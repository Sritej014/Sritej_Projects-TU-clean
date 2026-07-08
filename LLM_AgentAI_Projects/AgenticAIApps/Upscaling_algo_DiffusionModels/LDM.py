import os
import sys
import zipfile
import gc
import glob
import cv2
import torch
import numpy as np
import urllib.request
import subprocess
import tempfile
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from tqdm import tqdm

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer

# =====================================================
# CONFIGURATION
# =====================================================

ZIP_PATH = "/mnt/d/Dev/LLM_AgentAI_Projects/AgenticAIApps/LDM/kim.zip"

EXTRACT_TO = "extracted_photos"
OUTPUT_FOLDER = "restored_outputs"

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

TARGET_WIDTH = 1920   # set None if you only want native 4x output

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -----------------------------
# MODEL / TOOL TOGGLES
# -----------------------------
USE_GFPGAN = False
USE_CODEFORMER = True   # enable only if repo is installed and working
USE_SUPIR = False        # enable only after SUPIR standalone works

# Do not enable both face restorers together unless you really want to chain them
if USE_GFPGAN and USE_CODEFORMER:
    raise ValueError("Enable either GFPGAN or CodeFormer, not both at the same time.")

# -----------------------------
# REAL-ESRGAN
# -----------------------------
REALESRGAN_MODEL_PATH = "RealESRGAN_x4plus.pth"
REALESRGAN_MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
)

# -----------------------------
# GFPGAN
# -----------------------------
GFPGAN_MODEL_PATH = "GFPGANv1.4.pth"
GFPGAN_MODEL_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
)

# -----------------------------
# CODEFORMER
# -----------------------------
CODEFORMER_REPO = "/mnt/d/Dev/LLM_AgentAI_Projects/AgenticAIApps/CodeFormer"
CODEFORMER_INFER_SCRIPT = os.path.join(CODEFORMER_REPO, "inference_codeformer.py")
CODEFORMER_WEIGHT = os.path.join(
    CODEFORMER_REPO, "weights", "CodeFormer", "codeformer.pth"
)

# Fidelity: 0 = stronger restoration, 1 = more faithful
CODEFORMER_FIDELITY = 0.7

# -----------------------------
# SUPIR
# IMPORTANT:
# Replace this command template with the actual working SUPIR command in your setup.
# The placeholders {input} and {output_dir} will be replaced automatically.
# -----------------------------
SUPIR_CMD_TEMPLATE = [
    sys.executable,
    "/mnt/d/Dev/LLM_AgentAI_Projects/AgenticAIApps/SUPIR/YOUR_SUPIR_SCRIPT.py",
    "--input", "{input}",
    "--output", "{output_dir}",
]

# =====================================================
# SETUP
# =====================================================

os.makedirs(EXTRACT_TO, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

if os.path.exists(ZIP_PATH):
    if not os.listdir(EXTRACT_TO):
        print("Extracting zip...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_TO)
else:
    print("ZIP file not found. Continuing with existing extracted_photos folder.")

# =====================================================
# HELPERS
# =====================================================

def download_if_missing(url, path):
    if not os.path.exists(path):
        print(f"Downloading: {os.path.basename(path)}")
        urllib.request.urlretrieve(url, path)

def pil_to_bgr(img):
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def bgr_to_pil(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def resize_to_target_width(img, target_width):
    if target_width is None:
        return img

    w, h = img.size
    if w >= target_width:
        return img

    aspect_ratio = h / w
    target_height = int(target_width * aspect_ratio)
    return img.resize((target_width, target_height), Image.LANCZOS)

def find_generated_image(folder):
    candidates = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        candidates.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

# =====================================================
# LOAD MODELS
# =====================================================

download_if_missing(REALESRGAN_MODEL_URL, REALESRGAN_MODEL_PATH)

realesrgan_model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4
)

upsampler = RealESRGANer(
    scale=4,
    model_path=REALESRGAN_MODEL_PATH,
    model=realesrgan_model,
    tile=128,
    tile_pad=10,
    pre_pad=0,
    half=False,
    gpu_id=0 if device == "cuda" else None
)

gfpganer = None
if USE_GFPGAN:
    download_if_missing(GFPGAN_MODEL_URL, GFPGAN_MODEL_PATH)

    gfpganer = GFPGANer(
        model_path=GFPGAN_MODEL_PATH,
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None
    )

# =====================================================
# IMAGE ENHANCEMENT BRANCHES
# =====================================================

def enhance_faithful(img):
    """
    Conservative branch:
    closer to original, less hallucination
    """
    img = ImageOps.exif_transpose(img).convert("RGB")
    img_np = np.array(img)

    # Mild CLAHE on brightness channel
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Mild gamma lift
    gamma = 0.88
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    img_np = cv2.LUT(img_np, table)

    # Mild denoise
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.fastNlMeansDenoisingColored(img_bgr, None, 3, 3, 7, 21)
    img_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Gentle luminance sharpening
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    blurred_l = cv2.GaussianBlur(l, (0, 0), sigmaX=0.8)
    sharp_l = cv2.addWeighted(l, 1.22, blurred_l, -0.22, 0)
    sharp_l = np.clip(sharp_l, 0, 255).astype(np.uint8)

    lab = cv2.merge((sharp_l, a, b))
    img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    img = Image.fromarray(img_np)
    img = ImageEnhance.Brightness(img).enhance(1.05)
    img = ImageEnhance.Contrast(img).enhance(1.08)
    img = ImageEnhance.Color(img).enhance(1.03)
    img = ImageEnhance.Sharpness(img).enhance(1.05)

    img = img.filter(
        ImageFilter.UnsharpMask(
            radius=0.8,
            percent=70,
            threshold=4
        )
    )

    return img

def finish_ai_restored(img):
    """
    Stronger 'nice-looking' finish for the AI branch
    """
    img = ImageOps.exif_transpose(img).convert("RGB")
    img_np = np.array(img)

    # Local contrast
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Slight gamma lift
    gamma = 0.82
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    img_np = cv2.LUT(img_np, table)

    # Mild denoise
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.fastNlMeansDenoisingColored(img_bgr, None, 4, 4, 7, 21)
    img_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Mild luminance sharpening
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    blurred_l = cv2.GaussianBlur(l, (0, 0), sigmaX=1.0)
    sharp_l = cv2.addWeighted(l, 1.32, blurred_l, -0.32, 0)
    sharp_l = np.clip(sharp_l, 0, 255).astype(np.uint8)

    lab = cv2.merge((sharp_l, a, b))
    img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    img = Image.fromarray(img_np)
    img = ImageEnhance.Brightness(img).enhance(1.08)
    img = ImageEnhance.Contrast(img).enhance(1.12)
    img = ImageEnhance.Color(img).enhance(1.04)
    img = ImageEnhance.Sharpness(img).enhance(1.12)

    img = img.filter(
        ImageFilter.UnsharpMask(
            radius=1.0,
            percent=100,
            threshold=3
        )
    )

    return img

# =====================================================
# OPTIONAL AI MODULES
# =====================================================

def restore_faces_with_gfpgan(img):
    if gfpganer is None:
        return img

    img_bgr = pil_to_bgr(img)

    # paste_back=True returns the full image with restored faces blended in
    _, _, restored_bgr = gfpganer.enhance(
        img_bgr,
        has_aligned=False,
        only_center_face=False,
        paste_back=True
    )

    return bgr_to_pil(restored_bgr)

def restore_faces_with_codeformer(img):
    """
    Calls the official CodeFormer inference script externally.
    You must have CodeFormer repo installed and weights available.
    """
    if not USE_CODEFORMER:
        return img

    if not os.path.exists(CODEFORMER_INFER_SCRIPT):
        print("CodeFormer script not found. Skipping CodeFormer.")
        return img

    if not os.path.exists(CODEFORMER_WEIGHT):
        print("CodeFormer weight not found. Skipping CodeFormer.")
        return img

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.png")
        out_dir = os.path.join(tmpdir, "codeformer_out")
        os.makedirs(out_dir, exist_ok=True)

        img.save(input_path)

        cmd = [
            sys.executable,
            CODEFORMER_INFER_SCRIPT,
            "-i", input_path,
            "-o", out_dir,
            "-w", str(CODEFORMER_FIDELITY),
            "--upscale", "1"
        ]

        print("Running CodeFormer...")
        try:
            subprocess.run(cmd, cwd=CODEFORMER_REPO, check=True)
        except subprocess.CalledProcessError as e:
            print("CodeFormer failed:", e)
            return img

        result_path = find_generated_image(out_dir)
        if result_path is None:
            print("CodeFormer did not generate an output image.")
            return img

        return Image.open(result_path).convert("RGB")

def restore_with_supir(img):
    """
    Calls SUPIR externally.
    Replace SUPIR_CMD_TEMPLATE with your real working SUPIR command.
    """
    if not USE_SUPIR:
        return img

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.png")
        out_dir = os.path.join(tmpdir, "supir_out")
        os.makedirs(out_dir, exist_ok=True)

        img.save(input_path)

        cmd = [part.format(input=input_path, output_dir=out_dir) for part in SUPIR_CMD_TEMPLATE]

        print("Running SUPIR...")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print("SUPIR failed:", e)
            return img
        except FileNotFoundError as e:
            print("SUPIR command not found / wrong script path:", e)
            return img

        result_path = find_generated_image(out_dir)
        if result_path is None:
            print("SUPIR did not generate an output image.")
            return img

        return Image.open(result_path).convert("RGB")

# =====================================================
# FIND IMAGES
# =====================================================

image_paths = []

for ext in VALID_EXTENSIONS:
    image_paths.extend(glob.glob(os.path.join(EXTRACT_TO, "**", f"*{ext}"), recursive=True))
    image_paths.extend(glob.glob(os.path.join(EXTRACT_TO, "**", f"*{ext.upper()}"), recursive=True))

image_paths = sorted(list(set(image_paths)))

print(f"Found {len(image_paths)} images")

if len(image_paths) == 0:
    raise RuntimeError("No images found inside extracted_photos.")

# =====================================================
# MAIN LOOP
# =====================================================

for i, img_path in enumerate(tqdm(image_paths, desc="Restoring images"), start=1):
    filename = os.path.basename(img_path)
    name, ext = os.path.splitext(filename)

    output_faithful = os.path.join(OUTPUT_FOLDER, f"{name}_faithful.png")
    output_ai = os.path.join(OUTPUT_FOLDER, f"{name}_ai_restored.png")

    if os.path.exists(output_faithful) and os.path.exists(output_ai):
        print(f"Skipping already done: {filename}")
        continue

    try:
        print(f"\n[{i}/{len(image_paths)}] Processing: {filename}")

        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img).convert("RGB")

        original_w, original_h = img.size
        print(f"Original size: {original_w}x{original_h}")

        # -------------------------------------------------
        # Base super-resolution with Real-ESRGAN
        # -------------------------------------------------
        img_bgr = pil_to_bgr(img)

        with torch.inference_mode():
            output_bgr, _ = upsampler.enhance(img_bgr, outscale=4)

        sr_img = bgr_to_pil(output_bgr)
        print(f"After Real-ESRGAN: {sr_img.size[0]}x{sr_img.size[1]}")

        # -------------------------------------------------
        # Faithful output
        # -------------------------------------------------
        faithful_img = enhance_faithful(sr_img)
        faithful_img = resize_to_target_width(faithful_img, TARGET_WIDTH)
        faithful_img.save(output_faithful, quality=95)
        print(f"Saved faithful: {output_faithful}")

        # -------------------------------------------------
        # AI-restored output
        # Order:
        # Real-ESRGAN -> GFPGAN or CodeFormer -> SUPIR -> finish
        # -------------------------------------------------
        ai_img = sr_img.copy()

        if USE_GFPGAN:
            ai_img = restore_faces_with_gfpgan(ai_img)

        if USE_CODEFORMER:
            ai_img = restore_faces_with_codeformer(ai_img)

        if USE_SUPIR:
            ai_img = restore_with_supir(ai_img)

        ai_img = finish_ai_restored(ai_img)
        ai_img = resize_to_target_width(ai_img, TARGET_WIDTH)
        ai_img.save(output_ai, quality=95)
        print(f"Saved AI-restored: {output_ai}")

        del img
        del img_bgr
        del output_bgr
        del sr_img
        del faithful_img
        del ai_img

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    except RuntimeError as e:
        print(f"Runtime error on {filename}: {e}")
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        continue

    except Exception as e:
        print(f"Error on {filename}: {e}")
        gc.collect()
        continue

print("\nDone.")