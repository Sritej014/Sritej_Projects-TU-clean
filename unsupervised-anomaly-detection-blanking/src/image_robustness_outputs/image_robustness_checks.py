import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageStat
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, average_precision_score


IMAGE_EXTENSIONS = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
IMAGE_SIZE = 224
BATCH_SIZE = 32


# ============================================================
# DATASET HELPERS
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
# IMAGE STATISTICS
# ============================================================

def compute_color_stats(df, output_dir, modality_name):
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{modality_name}: color stats"):
        img = Image.open(row["path"]).convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0

        gray = np.asarray(ImageOps.grayscale(img)).astype(np.float32) / 255.0

        stats = {
            **row.to_dict(),
            "width": img.width,
            "height": img.height,

            "gray_mean": gray.mean(),
            "gray_std": gray.std(),
            "gray_min": gray.min(),
            "gray_max": gray.max(),

            "r_mean": arr[:, :, 0].mean(),
            "g_mean": arr[:, :, 1].mean(),
            "b_mean": arr[:, :, 2].mean(),

            "r_std": arr[:, :, 0].std(),
            "g_std": arr[:, :, 1].std(),
            "b_std": arr[:, :, 2].std(),
        }

        rows.append(stats)

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(os.path.join(output_dir, "image_color_brightness_stats.csv"), index=False)

    summary = stats_df.groupby("tool_state")[
        ["gray_mean", "gray_std", "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std"]
    ].agg(["mean", "std", "min", "median", "max"])

    summary.to_csv(os.path.join(output_dir, "image_color_brightness_summary.csv"))

    print(f"\n{modality_name} brightness/color summary:")
    print(summary)

    for col in ["gray_mean", "gray_std", "r_mean", "g_mean", "b_mean"]:
        plt.figure(figsize=(10, 6))

        plt.hist(stats_df.loc[stats_df["tool_state"] == "new", col], bins=40, alpha=0.6, label="new")
        plt.hist(stats_df.loc[stats_df["tool_state"] == "wear", col], bins=40, alpha=0.6, label="wear")

        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(f"{modality_name}: {col} comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col}_hist.png"), dpi=200)
        plt.close()

    return stats_df


def compute_grayscale_histograms(df, output_dir, modality_name, bins=64):
    rows = []
    histograms = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{modality_name}: grayscale hist"):
        img = Image.open(row["path"]).convert("L")
        arr = np.asarray(img).astype(np.float32) / 255.0

        hist, bin_edges = np.histogram(arr, bins=bins, range=(0, 1), density=True)

        histograms.append(hist)

        rows.append({
            "sample_id": row["sample_id"],
            "tool_state": row["tool_state"],
            "label": row["label"],
            "path": row["path"],
        })

    histograms = np.vstack(histograms)
    meta = pd.DataFrame(rows)

    np.save(os.path.join(output_dir, "grayscale_histograms.npy"), histograms)
    meta.to_csv(os.path.join(output_dir, "grayscale_histogram_metadata.csv"), index=False)

    # Plot mean grayscale histograms
    plt.figure(figsize=(10, 6))

    x = np.linspace(0, 1, bins)

    for state in ["new", "wear"]:
        mask = meta["tool_state"].values == state
        mean_hist = histograms[mask].mean(axis=0)
        std_hist = histograms[mask].std(axis=0)

        plt.plot(x, mean_hist, label=f"{state} mean")
        plt.fill_between(x, mean_hist - std_hist, mean_hist + std_hist, alpha=0.2)

    plt.xlabel("Grayscale intensity")
    plt.ylabel("Density")
    plt.title(f"{modality_name}: mean grayscale histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mean_grayscale_histogram.png"), dpi=200)
    plt.close()

    # Anomaly detection on grayscale histograms
    X_new = histograms[meta["tool_state"].values == "new"]
    X_all = histograms

    scaler = StandardScaler()
    X_new_s = scaler.fit_transform(X_new)
    X_all_s = scaler.transform(X_all)

    results = meta.copy()

    pca = PCA(n_components=0.95)
    pca.fit(X_new_s)
    X_rec = pca.inverse_transform(pca.transform(X_all_s))
    results["grayhist_pca_score"] = np.mean((X_all_s - X_rec) ** 2, axis=1)

    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=42)
    iso.fit(X_new_s)
    results["grayhist_if_score"] = -iso.score_samples(X_all_s)

    knn = NearestNeighbors(n_neighbors=min(5, len(X_new_s)))
    knn.fit(X_new_s)
    distances, _ = knn.kneighbors(X_all_s)
    results["grayhist_knn_score"] = distances.mean(axis=1)

    metrics = evaluate_scores(results, ["grayhist_pca_score", "grayhist_if_score", "grayhist_knn_score"])
    metrics.to_csv(os.path.join(output_dir, "grayscale_histogram_model_metrics.csv"), index=False)
    results.to_csv(os.path.join(output_dir, "grayscale_histogram_scores.csv"), index=False)

    print(f"\n{modality_name} grayscale histogram model metrics:")
    print(metrics)

    return results, metrics


# ============================================================
# FEATURE EXTRACTOR
# ============================================================

class ResNet18FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.transform = weights.transforms()

    def forward(self, x):
        features = self.feature_extractor(x)
        return torch.flatten(features, 1)


def preprocess_image(path, mode):
    img = Image.open(path).convert("RGB")
    w, h = img.size

    if mode == "full":
        return img

    if mode == "center_crop":
        crop_size = int(min(w, h) * 0.65)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        img = img.crop((left, top, left + crop_size, top + crop_size))
        return img

    if mode == "large_center_crop":
        crop_size = int(min(w, h) * 0.85)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        img = img.crop((left, top, left + crop_size, top + crop_size))
        return img

    if mode == "random_crop_fixed":
        # deterministic pseudo-random crop based on filename hash
        rng = np.random.default_rng(abs(hash(path)) % (2**32))
        crop_size = int(min(w, h) * 0.75)

        max_left = w - crop_size
        max_top = h - crop_size

        left = int(rng.integers(0, max_left + 1)) if max_left > 0 else 0
        top = int(rng.integers(0, max_top + 1)) if max_top > 0 else 0

        img = img.crop((left, top, left + crop_size, top + crop_size))
        return img

    raise ValueError(f"Unknown mode: {mode}")


def extract_resnet_embeddings(df, output_dir, modality_name, mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18FeatureExtractor().to(device)
    model.eval()

    embeddings = []

    with torch.no_grad():
        for start in tqdm(range(0, len(df), BATCH_SIZE), desc=f"{modality_name}: {mode} embeddings"):
            batch = df.iloc[start:start + BATCH_SIZE]

            imgs = []
            for path in batch["path"]:
                img = preprocess_image(path, mode)
                tensor = model.transform(img)
                imgs.append(tensor)

            x = torch.stack(imgs).to(device)
            feat = model(x).cpu().numpy()
            embeddings.append(feat)

    embeddings = np.vstack(embeddings)
    np.save(os.path.join(output_dir, f"resnet18_embeddings_{mode}.npy"), embeddings)

    return embeddings


# ============================================================
# MODELS / EVAL
# ============================================================

def evaluate_scores(results_df, score_cols):
    y = results_df["label"].values.astype(int)
    rows = []

    for col in score_cols:
        scores = results_df[col].values

        rows.append({
            "score": col,
            "roc_auc": roc_auc_score(y, scores),
            "pr_auc": average_precision_score(y, scores),
            "new_mean_score": results_df.loc[results_df["tool_state"] == "new", col].mean(),
            "wear_mean_score": results_df.loc[results_df["tool_state"] == "wear", col].mean(),
            "wear_minus_new_mean": (
                results_df.loc[results_df["tool_state"] == "wear", col].mean()
                - results_df.loc[results_df["tool_state"] == "new", col].mean()
            ),
        })

    return pd.DataFrame(rows)


def run_embedding_anomaly_models(embeddings, df, output_dir, mode):
    X_new = embeddings[df["tool_state"].values == "new"]
    X_all = embeddings

    scaler = StandardScaler()
    X_new_s = scaler.fit_transform(X_new)
    X_all_s = scaler.transform(X_all)

    results = df.copy()

    pca = PCA(n_components=0.95)
    pca.fit(X_new_s)
    X_rec = pca.inverse_transform(pca.transform(X_all_s))
    results[f"{mode}_pca_score"] = np.mean((X_all_s - X_rec) ** 2, axis=1)

    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=42)
    iso.fit(X_new_s)
    results[f"{mode}_if_score"] = -iso.score_samples(X_all_s)

    knn = NearestNeighbors(n_neighbors=min(5, len(X_new_s)))
    knn.fit(X_new_s)
    distances, _ = knn.kneighbors(X_all_s)
    results[f"{mode}_knn_score"] = distances.mean(axis=1)

    score_cols = [f"{mode}_pca_score", f"{mode}_if_score", f"{mode}_knn_score"]
    metrics = evaluate_scores(results, score_cols)
    metrics["crop_mode"] = mode

    results.to_csv(os.path.join(output_dir, f"embedding_scores_{mode}.csv"), index=False)
    metrics.to_csv(os.path.join(output_dir, f"embedding_metrics_{mode}.csv"), index=False)

    # PCA 2D visual
    Z = PCA(n_components=2).fit_transform(X_all_s)

    plt.figure(figsize=(8, 7))
    for state in ["new", "wear"]:
        mask = results["tool_state"].values == state
        plt.scatter(Z[mask, 0], Z[mask, 1], s=15, alpha=0.7, label=state)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"2D PCA embeddings: {mode}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"embedding_pca_2d_{mode}.png"), dpi=200)
    plt.close()

    # Histograms
    for col in score_cols:
        plt.figure(figsize=(10, 6))
        plt.hist(results.loc[results["tool_state"] == "new", col], bins=40, alpha=0.6, label="new")
        plt.hist(results.loc[results["tool_state"] == "wear", col], bins=40, alpha=0.6, label="wear")
        plt.xlabel("Anomaly score")
        plt.ylabel("Count")
        plt.title(f"{col}: new vs wear")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col}_hist.png"), dpi=200)
        plt.close()

    return results, metrics


# ============================================================
# MODALITY RUNNER
# ============================================================

def run_modality(image_root, modality_name, output_root):
    modality_dir = os.path.join(image_root, modality_name)
    output_dir = os.path.join(output_root, modality_name)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 100)
    print(f"IMAGE ROBUSTNESS CHECKS: {modality_name}")
    print("=" * 100)

    df = build_image_table(modality_dir, modality_name)
    df.to_csv(os.path.join(output_dir, "image_table.csv"), index=False)

    print(df.groupby("tool_state").size())

    # 1. Brightness/color statistics
    compute_color_stats(df, output_dir, modality_name)

    # 2. Grayscale histogram comparison
    _, gray_metrics = compute_grayscale_histograms(df, output_dir, modality_name)

    all_metrics = []

    # 3/4. Full vs center crop vs large center crop vs random crop
    crop_modes = [
        "full",
        "large_center_crop",
        "center_crop",
        "random_crop_fixed",
    ]

    for mode in crop_modes:
        embeddings = extract_resnet_embeddings(df, output_dir, modality_name, mode)
        _, metrics = run_embedding_anomaly_models(embeddings, df, output_dir, mode)
        all_metrics.append(metrics)

    all_metrics = pd.concat(all_metrics, ignore_index=True)
    all_metrics.to_csv(os.path.join(output_dir, "crop_robustness_embedding_metrics.csv"), index=False)

    gray_metrics["crop_mode"] = "grayscale_histogram_only"
    gray_metrics = gray_metrics.rename(columns={"score": "metric_name"})

    embedding_metrics = all_metrics.rename(columns={"score": "metric_name"})
    combined = pd.concat([embedding_metrics, gray_metrics], ignore_index=True)
    combined["modality"] = modality_name

    combined.to_csv(os.path.join(output_dir, "combined_robustness_metrics.csv"), index=False)

    print(f"\n{modality_name} combined robustness metrics:")
    print(combined.to_string(index=False))

    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", default="Images")
    parser.add_argument("--output_root", default="image_robustness_outputs")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    all_combined = []

    for modality_name in ["Punch", "Sheet"]:
        combined = run_modality(args.image_root, modality_name, args.output_root)
        all_combined.append(combined)

    summary = pd.concat(all_combined, ignore_index=True)
    summary.to_csv(os.path.join(args.output_root, "image_robustness_summary.csv"), index=False)

    print("\n" + "=" * 100)
    print("FINAL IMAGE ROBUSTNESS SUMMARY")
    print("=" * 100)
    print(summary.to_string(index=False))
    print(f"\nSaved: {os.path.join(args.output_root, 'image_robustness_summary.csv')}")


if __name__ == "__main__":
    main()
