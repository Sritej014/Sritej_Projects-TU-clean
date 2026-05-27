import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.metrics import roc_auc_score, average_precision_score


IMAGE_EXTENSIONS = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]


# ============================================================
# DATA HELPERS
# ============================================================

def find_images(folder):
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
        paths.extend(glob.glob(os.path.join(folder, ext.upper())))
    return sorted(paths)


def build_image_table(modality_dir, modality_name):
    rows = []

    for state, label in [("new", 0), ("wear", 1)]:
        folder = os.path.join(modality_dir, state)
        paths = find_images(folder)

        print(f"{modality_name} / {state}: {len(paths)} images")

        for i, path in enumerate(paths):
            rows.append({
                "sample_id": f"{modality_name}_{state}_{i:06d}",
                "modality": modality_name,
                "tool_state": state,
                "label": label,
                "path": path,
                "filename": os.path.basename(path),
            })

    return pd.DataFrame(rows)


# ============================================================
# PATCH FEATURE EXTRACTOR
# ============================================================

class ResNetPatchExtractor(nn.Module):
    """
    PatchCore-style feature extractor using intermediate ResNet18 feature maps.
    We use layer2 and layer3 features, upsample layer3 to layer2 resolution,
    concatenate them, and treat each spatial location as a patch embedding.
    """

    def __init__(self):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT
        resnet = models.resnet18(weights=weights)

        self.transform = weights.transforms()

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)

        feat2 = self.layer2(x)  # B x C2 x H2 x W2
        feat3 = self.layer3(feat2)  # B x C3 x H3 x W3

        feat3_up = F.interpolate(
            feat3,
            size=feat2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        features = torch.cat([feat2, feat3_up], dim=1)

        return features


def load_image_tensor(path, transform):
    img = Image.open(path).convert("RGB")
    return transform(img)


def extract_patch_embeddings(df, model, device, batch_size=16):
    """
    Returns:
    patch_embeddings: N_total_patches x D
    image_patch_features: list of patch feature arrays per image
    spatial_shape: H_patch, W_patch
    """

    model.eval()

    all_patches = []
    per_image_patches = []
    spatial_shape = None

    with torch.no_grad():
        for start in tqdm(range(0, len(df), batch_size), desc="Extracting patch embeddings"):
            batch_df = df.iloc[start:start + batch_size]

            imgs = []
            for path in batch_df["path"]:
                tensor = load_image_tensor(path, model.transform)
                imgs.append(tensor)

            x = torch.stack(imgs).to(device)
            feats = model(x)

            # B x C x H x W
            B, C, H, W = feats.shape
            spatial_shape = (H, W)

            # B x H x W x C
            feats = feats.permute(0, 2, 3, 1).contiguous()

            for b in range(B):
                patches = feats[b].reshape(H * W, C).cpu().numpy()
                per_image_patches.append(patches)
                all_patches.append(patches)

    all_patches = np.vstack(all_patches)

    return all_patches, per_image_patches, spatial_shape


# ============================================================
# MEMORY BANK
# ============================================================

def build_memory_bank(train_patches, max_memory_patches=20000, random_state=42):
    """
    Simple random memory reduction.
    Full PatchCore often uses coreset sampling, but random reduction is good first baseline.
    """

    rng = np.random.default_rng(random_state)

    n = train_patches.shape[0]

    if n <= max_memory_patches:
        return train_patches

    idx = rng.choice(n, size=max_memory_patches, replace=False)
    return train_patches[idx]


def compute_patchcore_scores(per_image_patches, memory_bank, device, batch_size=4096):
    """
    For each image:
    - compute nearest memory-bank distance for every patch
    - image score = max patch distance
    - heatmap = patch distance map
    """

    memory = torch.tensor(memory_bank, dtype=torch.float32).to(device)

    image_scores = []
    image_patch_scores = []

    for patches in tqdm(per_image_patches, desc="Scoring images"):
        patches_t = torch.tensor(patches, dtype=torch.float32).to(device)

        distances_all = []

        for start in range(0, patches_t.shape[0], batch_size):
            chunk = patches_t[start:start + batch_size]

            # distance: chunk_patches x memory_bank
            dists = torch.cdist(chunk, memory)
            min_dists = torch.min(dists, dim=1).values
            distances_all.append(min_dists.detach().cpu())

        patch_scores = torch.cat(distances_all).numpy()

        image_score = float(np.max(patch_scores))

        image_scores.append(image_score)
        image_patch_scores.append(patch_scores)

    return np.array(image_scores), image_patch_scores


# ============================================================
# HEATMAP VISUALIZATION
# ============================================================

def save_heatmap_overlay(image_path, patch_scores, spatial_shape, output_path):
    img = Image.open(image_path).convert("RGB")
    img_arr = np.asarray(img).astype(np.float32) / 255.0

    H, W = spatial_shape
    heat = patch_scores.reshape(H, W)

    # Normalize heatmap
    heat = heat - heat.min()
    if heat.max() > 1e-12:
        heat = heat / heat.max()

    heat_t = torch.tensor(heat)[None, None, :, :].float()
    heat_up = F.interpolate(
        heat_t,
        size=(img_arr.shape[0], img_arr.shape[1]),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(img_arr)
    plt.imshow(heat_up, alpha=0.45, cmap="jet")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_top_heatmaps(results, image_patch_scores, spatial_shape, output_dir, method_name="patchcore", top_n=20):
    heatmap_dir = os.path.join(output_dir, "top_heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    top = results.sort_values("patchcore_score", ascending=False).head(top_n)

    for rank, (idx, row) in enumerate(top.iterrows(), start=1):
        output_path = os.path.join(
            heatmap_dir,
            f"{rank:02d}_{row['tool_state']}_{row['sample_id']}.png"
        )

        save_heatmap_overlay(
            image_path=row["path"],
            patch_scores=image_patch_scores[idx],
            spatial_shape=spatial_shape,
            output_path=output_path,
        )


def save_contact_sheet_of_heatmaps(results, image_patch_scores, spatial_shape, output_path, top_n=12):
    top = results.sort_values("patchcore_score", ascending=False).head(top_n)

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.flatten()

    for ax, (idx, row) in zip(axes, top.iterrows()):
        img = Image.open(row["path"]).convert("RGB")
        img_arr = np.asarray(img).astype(np.float32) / 255.0

        H, W = spatial_shape
        heat = image_patch_scores[idx].reshape(H, W)
        heat = heat - heat.min()
        if heat.max() > 1e-12:
            heat = heat / heat.max()

        heat_t = torch.tensor(heat)[None, None, :, :].float()
        heat_up = F.interpolate(
            heat_t,
            size=(img_arr.shape[0], img_arr.shape[1]),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()

        ax.imshow(img_arr)
        ax.imshow(heat_up, alpha=0.45, cmap="jet")
        ax.set_title(f"{row['tool_state']} | {row['patchcore_score']:.2f}", fontsize=8)
        ax.axis("off")

    for ax in axes[len(top):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ============================================================
# EVALUATION / PLOTS
# ============================================================

def evaluate(results):
    y = results["label"].values.astype(int)
    scores = results["patchcore_score"].values

    metrics = pd.DataFrame([{
        "method": "patchcore_resnet18_random_memory",
        "roc_auc": roc_auc_score(y, scores),
        "pr_auc": average_precision_score(y, scores),
        "new_mean_score": results.loc[results["tool_state"] == "new", "patchcore_score"].mean(),
        "wear_mean_score": results.loc[results["tool_state"] == "wear", "patchcore_score"].mean(),
        "wear_minus_new_mean": (
            results.loc[results["tool_state"] == "wear", "patchcore_score"].mean()
            - results.loc[results["tool_state"] == "new", "patchcore_score"].mean()
        )
    }])

    return metrics


def plot_score_hist(results, output_dir):
    plt.figure(figsize=(10, 6))

    plt.hist(
        results.loc[results["tool_state"] == "new", "patchcore_score"],
        bins=40,
        alpha=0.6,
        label="new",
    )

    plt.hist(
        results.loc[results["tool_state"] == "wear", "patchcore_score"],
        bins=40,
        alpha=0.6,
        label="wear",
    )

    plt.xlabel("PatchCore anomaly score")
    plt.ylabel("Count")
    plt.title("PatchCore image-level score: new vs wear")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "patchcore_score_hist.png"), dpi=200)
    plt.close()


# ============================================================
# RUN MODALITY
# ============================================================

def run_modality(image_root, modality_name, output_root, max_memory_patches, batch_size):
    modality_dir = os.path.join(image_root, modality_name)
    output_dir = os.path.join(output_root, modality_name)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 100)
    print(f"PATCHCORE BASELINE: {modality_name}")
    print("=" * 100)

    df = build_image_table(modality_dir, modality_name)
    df.to_csv(os.path.join(output_dir, "patchcore_image_table.csv"), index=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = ResNetPatchExtractor().to(device)

    df_new = df[df["tool_state"] == "new"].reset_index(drop=True)
    df_all = df.reset_index(drop=True)

    print("\nExtracting training/new patch embeddings...")
    train_patches, _, spatial_shape = extract_patch_embeddings(
        df_new,
        model,
        device,
        batch_size=batch_size,
    )

    print("Raw train patches:", train_patches.shape)
    print("Patch spatial shape:", spatial_shape)

    memory_bank = build_memory_bank(
        train_patches,
        max_memory_patches=max_memory_patches,
    )

    print("Memory bank:", memory_bank.shape)
    np.save(os.path.join(output_dir, "memory_bank.npy"), memory_bank)

    print("\nExtracting all image patch embeddings...")
    _, per_image_patches, spatial_shape_all = extract_patch_embeddings(
        df_all,
        model,
        device,
        batch_size=batch_size,
    )

    print("\nScoring all images...")
    image_scores, image_patch_scores = compute_patchcore_scores(
        per_image_patches,
        memory_bank,
        device,
    )

    results = df_all.copy()
    results["patchcore_score"] = image_scores

    results.to_csv(os.path.join(output_dir, "patchcore_scores.csv"), index=False)

    metrics = evaluate(results)
    metrics.to_csv(os.path.join(output_dir, "patchcore_metrics.csv"), index=False)

    print("\nPatchCore metrics:")
    print(metrics)

    plot_score_hist(results, output_dir)

    print("\nSaving top anomaly heatmaps...")
    save_top_heatmaps(
        results,
        image_patch_scores,
        spatial_shape_all,
        output_dir,
        top_n=20,
    )

    save_contact_sheet_of_heatmaps(
        results,
        image_patch_scores,
        spatial_shape_all,
        os.path.join(output_dir, "top_heatmaps_contact_sheet.png"),
        top_n=12,
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", default="Images")
    parser.add_argument("--output_root", default="patchcore_outputs")
    parser.add_argument("--max_memory_patches", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    all_metrics = []

    for modality_name in ["Punch", "Sheet"]:
        metrics = run_modality(
            args.image_root,
            modality_name,
            args.output_root,
            args.max_memory_patches,
            args.batch_size,
        )
        metrics["modality"] = modality_name
        all_metrics.append(metrics)

    summary = pd.concat(all_metrics, ignore_index=True)
    summary = summary[
        [
            "modality",
            "method",
            "roc_auc",
            "pr_auc",
            "new_mean_score",
            "wear_mean_score",
            "wear_minus_new_mean",
        ]
    ]

    summary.to_csv(os.path.join(args.output_root, "patchcore_summary.csv"), index=False)

    print("\n" + "=" * 100)
    print("PATCHCORE SUMMARY")
    print("=" * 100)
    print(summary.to_string(index=False))
    print(f"\nSaved: {os.path.join(args.output_root, 'patchcore_summary.csv')}")


if __name__ == "__main__":
    main()
