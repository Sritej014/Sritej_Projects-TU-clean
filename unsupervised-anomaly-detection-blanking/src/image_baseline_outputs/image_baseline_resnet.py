import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from tqdm import tqdm
import joblib

import torch
import torch.nn as nn
from torchvision import models, transforms

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# CONFIG
# ============================================================

IMAGE_EXTENSIONS = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]

OUTPUT_ROOT = "image_baseline_outputs"
BATCH_SIZE = 32
IMAGE_SIZE = 224


# ============================================================
# DATA LOADING
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
# IMAGE AUDIT
# ============================================================

def audit_images(df, output_dir):
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Auditing images"):
        path = row["path"]

        try:
            img = Image.open(path)
            rows.append({
                **row.to_dict(),
                "valid": True,
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "error": "",
            })
        except Exception as e:
            rows.append({
                **row.to_dict(),
                "valid": False,
                "width": np.nan,
                "height": np.nan,
                "mode": "",
                "error": str(e),
            })

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(os.path.join(output_dir, "image_audit.csv"), index=False)

    print("\nImage audit summary:")
    print(audit_df.groupby(["tool_state", "valid"]).size())

    if audit_df["valid"].any():
        valid = audit_df[audit_df["valid"] == True]
        print("\nImage sizes:")
        print(valid.groupby("tool_state")[["width", "height"]].describe())

    return audit_df


def make_contact_sheet(df, output_path, max_per_state=20):
    selected = []

    for state in ["new", "wear"]:
        state_df = df[(df["tool_state"] == state) & (df["valid"] == True)].head(max_per_state)
        selected.extend(state_df.to_dict("records"))

    thumbs = []
    labels = []

    for row in selected:
        img = Image.open(row["path"]).convert("RGB")
        img.thumbnail((160, 160))
        canvas = Image.new("RGB", (160, 180), "white")
        canvas.paste(img, ((160 - img.width) // 2, 0))
        thumbs.append(canvas)
        labels.append(row["tool_state"])

    if not thumbs:
        return

    cols = 10
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 160, rows * 180), "white")

    for i, thumb in enumerate(thumbs):
        x = (i % cols) * 160
        y = (i // cols) * 180
        sheet.paste(thumb, (x, y))

    sheet.save(output_path)


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
        features = torch.flatten(features, 1)
        return features


def extract_embeddings(df, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model = ResNet18FeatureExtractor().to(device)
    model.eval()

    valid_df = df[df["valid"] == True].reset_index(drop=True).copy()

    embeddings = []

    with torch.no_grad():
        for start in tqdm(range(0, len(valid_df), BATCH_SIZE), desc="Extracting embeddings"):
            batch_df = valid_df.iloc[start:start + BATCH_SIZE]

            imgs = []
            for path in batch_df["path"]:
                img = Image.open(path).convert("RGB")
                img_tensor = model.transform(img)
                imgs.append(img_tensor)

            x = torch.stack(imgs).to(device)
            feat = model(x).cpu().numpy()
            embeddings.append(feat)

    embeddings = np.vstack(embeddings)

    np.save(os.path.join(output_dir, "resnet18_embeddings.npy"), embeddings)
    valid_df.to_csv(os.path.join(output_dir, "valid_image_metadata.csv"), index=False)

    print("Embeddings shape:", embeddings.shape)

    return embeddings, valid_df


# ============================================================
# ANOMALY MODELS
# ============================================================

def pca_reconstruction_score(X_train, X_all):
    pca = PCA(n_components=0.95)
    pca.fit(X_train)

    X_rec = pca.inverse_transform(pca.transform(X_all))
    scores = np.mean((X_all - X_rec) ** 2, axis=1)

    return pca, scores


def isolation_forest_score(X_train, X_all):
    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
    )
    model.fit(X_train)
    scores = -model.score_samples(X_all)
    return model, scores


def knn_distance_score(X_train, X_all, k=5):
    k = min(k, len(X_train))
    model = NearestNeighbors(n_neighbors=k)
    model.fit(X_train)

    distances, _ = model.kneighbors(X_all)
    scores = distances.mean(axis=1)

    return model, scores


# ============================================================
# EVALUATION / PLOTS
# ============================================================

def evaluate(results_df):
    y = results_df["label"].values.astype(int)

    rows = []
    for method in ["pca_score", "isolation_forest_score", "knn_score"]:
        scores = results_df[method].values

        rows.append({
            "method": method,
            "roc_auc": roc_auc_score(y, scores),
            "pr_auc": average_precision_score(y, scores),
            "new_mean_score": results_df.loc[results_df["tool_state"] == "new", method].mean(),
            "wear_mean_score": results_df.loc[results_df["tool_state"] == "wear", method].mean(),
            "wear_minus_new_mean": (
                results_df.loc[results_df["tool_state"] == "wear", method].mean()
                - results_df.loc[results_df["tool_state"] == "new", method].mean()
            )
        })

    return pd.DataFrame(rows)


def plot_histograms(results_df, output_dir):
    for method in ["pca_score", "isolation_forest_score", "knn_score"]:
        plt.figure(figsize=(10, 6))

        new_scores = results_df.loc[results_df["tool_state"] == "new", method]
        wear_scores = results_df.loc[results_df["tool_state"] == "wear", method]

        plt.hist(new_scores, bins=40, alpha=0.6, label="new")
        plt.hist(wear_scores, bins=40, alpha=0.6, label="wear")

        plt.xlabel("Anomaly score")
        plt.ylabel("Count")
        plt.title(f"{method}: new vs wear")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{method}_hist.png"), dpi=200)
        plt.close()


def plot_pca_2d(X_scaled, metadata, output_dir):
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 7))

    for state in ["new", "wear"]:
        mask = metadata["tool_state"] == state
        plt.scatter(Z[mask, 0], Z[mask, 1], s=15, alpha=0.7, label=state)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("2D PCA of ResNet18 image embeddings")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "embedding_pca_2d.png"), dpi=200)
    plt.close()


def save_top_anomaly_contact_sheet(results_df, method, output_path, top_n=20):
    top = results_df.sort_values(method, ascending=False).head(top_n)

    thumbs = []

    for _, row in top.iterrows():
        img = Image.open(row["path"]).convert("RGB")
        img.thumbnail((180, 180))
        canvas = Image.new("RGB", (180, 210), "white")
        canvas.paste(img, ((180 - img.width) // 2, 0))
        thumbs.append(canvas)

    if not thumbs:
        return

    cols = 5
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 180, rows * 210), "white")

    for i, thumb in enumerate(thumbs):
        x = (i % cols) * 180
        y = (i // cols) * 210
        sheet.paste(thumb, (x, y))

    sheet.save(output_path)


# ============================================================
# MAIN
# ============================================================

def run_modality(modality_dir, modality_name, output_root):
    output_dir = os.path.join(output_root, modality_name)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 100)
    print(f"IMAGE BASELINE: {modality_name}")
    print("=" * 100)

    df = build_image_table(modality_dir, modality_name)
    df.to_csv(os.path.join(output_dir, "image_table.csv"), index=False)

    audit_df = audit_images(df, output_dir)
    make_contact_sheet(
        audit_df,
        os.path.join(output_dir, "contact_sheet_new_wear.png")
    )

    embeddings, meta = extract_embeddings(audit_df, output_dir)

    X_new = embeddings[meta["tool_state"].values == "new"]
    X_all = embeddings

    scaler = StandardScaler()
    X_new_scaled = scaler.fit_transform(X_new)
    X_all_scaled = scaler.transform(X_all)

    joblib.dump(scaler, os.path.join(output_dir, "embedding_scaler.joblib"))

    print("\nTraining image PCA anomaly model...")
    pca_model, pca_scores = pca_reconstruction_score(X_new_scaled, X_all_scaled)

    print("Training image Isolation Forest...")
    if_model, if_scores = isolation_forest_score(X_new_scaled, X_all_scaled)

    print("Training image kNN distance model...")
    knn_model, knn_scores = knn_distance_score(X_new_scaled, X_all_scaled, k=5)

    results = meta.copy()
    results["pca_score"] = pca_scores
    results["isolation_forest_score"] = if_scores
    results["knn_score"] = knn_scores

    results.to_csv(os.path.join(output_dir, "image_model_scores.csv"), index=False)

    metrics = evaluate(results)
    metrics.to_csv(os.path.join(output_dir, "image_model_comparison_metrics.csv"), index=False)

    print("\nMetrics:")
    print(metrics)

    plot_histograms(results, output_dir)
    plot_pca_2d(X_all_scaled, results, output_dir)

    joblib.dump(pca_model, os.path.join(output_dir, "pca_model.joblib"))
    joblib.dump(if_model, os.path.join(output_dir, "isolation_forest_model.joblib"))
    joblib.dump(knn_model, os.path.join(output_dir, "knn_model.joblib"))

    for method in ["pca_score", "isolation_forest_score", "knn_score"]:
        top = results.sort_values(method, ascending=False).head(30)
        top.to_csv(os.path.join(output_dir, f"top_30_{method}.csv"), index=False)

        save_top_anomaly_contact_sheet(
            results,
            method,
            os.path.join(output_dir, f"top_anomalies_{method}.png"),
            top_n=20
        )

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", default="Images")
    parser.add_argument("--output_root", default=OUTPUT_ROOT)
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    modalities = {
        "Punch": os.path.join(args.image_root, "Punch"),
        "Sheet": os.path.join(args.image_root, "Sheet"),
    }

    all_metrics = []

    for modality_name, modality_dir in modalities.items():
        metrics = run_modality(modality_dir, modality_name, args.output_root)
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

    summary.to_csv(os.path.join(args.output_root, "image_baseline_summary.csv"), index=False)

    print("\n" + "=" * 100)
    print("IMAGE BASELINE SUMMARY")
    print("=" * 100)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
