import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nptdms import TdmsFile
from scipy.signal import resample
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CONFIG
# ============================================================

FORCE_CHANNELS = [
    "Modul7_Oberwerkzeug",
    "Modul10_Oberwerkzeug",
    "Modul7_Unterwerkzeug",
    "Modul10_Unterwerkzeug",
]

TRIGGER_CHANNEL = "Trigger"

RESAMPLED_LENGTH = 512
MIN_STROKE_LENGTH = 1000
MAX_STROKE_LENGTH = 20000

OUTPUT_DIR = "force_new_vs_wear_outputs"


# ============================================================
# UTILS
# ============================================================

def safe_area(x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(x)
    return np.trapz(x)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ============================================================
# TDMS READING
# ============================================================

def read_tdms_file(tdms_path):
    tdms_file = TdmsFile.read(tdms_path)

    groups = tdms_file.groups()
    if len(groups) == 0:
        raise ValueError(f"No groups found in {tdms_path}")

    group = groups[0]
    data = {}

    required_channels = FORCE_CHANNELS + [TRIGGER_CHANNEL]

    for ch in required_channels:
        if ch not in group:
            raise KeyError(f"Missing required channel {ch} in {tdms_path}")
        data[ch] = np.asarray(group[ch][:], dtype=float)

    return group.name, data


# ============================================================
# SEGMENTATION
# ============================================================

def get_trigger_segments(trigger):
    trigger = np.asarray(trigger)
    threshold = 0.5 * (np.nanmax(trigger) + np.nanmin(trigger))
    binary = trigger > threshold

    rising_edges = np.where(np.diff(binary.astype(int)) == 1)[0]

    segments = []
    for i in range(len(rising_edges) - 1):
        start = int(rising_edges[i])
        end = int(rising_edges[i + 1])
        segments.append((start, end))

    return segments, rising_edges


# ============================================================
# QUALITY CHECKS
# ============================================================

def check_stroke_quality(stroke_raw, raw_length):
    reasons = []

    if raw_length < MIN_STROKE_LENGTH:
        reasons.append("too_short")

    if raw_length > MAX_STROKE_LENGTH:
        reasons.append("too_long")

    if not np.all(np.isfinite(stroke_raw)):
        reasons.append("non_finite_values")

    # Check if all selected channels are nearly flat
    channel_ranges = np.ptp(stroke_raw, axis=0)
    if np.all(channel_ranges < 1e-6):
        reasons.append("all_channels_flat")

    # Check if any channel has unrealistic numerical explosion
    if np.nanmax(np.abs(stroke_raw)) > 1e6:
        reasons.append("unrealistic_amplitude")

    valid = len(reasons) == 0

    return valid, ";".join(reasons) if reasons else "valid"


# ============================================================
# STROKE EXTRACTION
# ============================================================

def extract_strokes_from_tdms(tdms_path, tool_state):
    group_name, data = read_tdms_file(tdms_path)

    trigger = data[TRIGGER_CHANNEL]
    segments, rising_edges = get_trigger_segments(trigger)

    force_matrix = np.vstack([data[ch] for ch in FORCE_CHANNELS]).T

    strokes_resampled = []
    metadata_rows = []

    for local_stroke_id, (start, end) in enumerate(segments):
        raw_length = end - start
        stroke_raw = force_matrix[start:end, :]

        valid, quality_reason = check_stroke_quality(stroke_raw, raw_length)

        row = {
            "global_sample_id": None,
            "tool_state": tool_state,
            "label": 0 if tool_state == "new" else 1,
            "source_file": os.path.basename(tdms_path),
            "group": group_name,
            "local_stroke_id": local_stroke_id,
            "raw_start": start,
            "raw_end": end,
            "raw_length": raw_length,
            "valid": valid,
            "quality_reason": quality_reason,
        }

        for c, ch in enumerate(FORCE_CHANNELS):
            x = stroke_raw[:, c] if raw_length > 0 else np.array([])

            if len(x) > 0:
                row[f"{ch}_raw_max"] = np.nanmax(x)
                row[f"{ch}_raw_min"] = np.nanmin(x)
                row[f"{ch}_raw_mean"] = np.nanmean(x)
                row[f"{ch}_raw_std"] = np.nanstd(x)
                row[f"{ch}_raw_range"] = np.nanmax(x) - np.nanmin(x)
                row[f"{ch}_raw_area"] = safe_area(x)
            else:
                row[f"{ch}_raw_max"] = np.nan
                row[f"{ch}_raw_min"] = np.nan
                row[f"{ch}_raw_mean"] = np.nan
                row[f"{ch}_raw_std"] = np.nan
                row[f"{ch}_raw_range"] = np.nan
                row[f"{ch}_raw_area"] = np.nan

        if valid:
            stroke_resampled = resample(stroke_raw, RESAMPLED_LENGTH, axis=0)

            # Baseline correction:
            # subtract pre-contact mean from each channel.
            baseline = np.mean(stroke_resampled[:80, :], axis=0, keepdims=True)
            stroke_resampled = stroke_resampled - baseline

            strokes_resampled.append(stroke_resampled)

        metadata_rows.append(row)

    metadata = pd.DataFrame(metadata_rows)

    if len(strokes_resampled) > 0:
        strokes_resampled = np.stack(strokes_resampled, axis=0)
    else:
        strokes_resampled = np.empty((0, RESAMPLED_LENGTH, len(FORCE_CHANNELS)))

    return strokes_resampled, metadata, len(rising_edges)


def load_dataset(folder, tool_state):
    tdms_files = sorted(glob.glob(os.path.join(folder, "*.tdms")))

    if len(tdms_files) == 0:
        raise FileNotFoundError(f"No TDMS files found in {folder}")

    all_strokes = []
    all_metadata = []

    print(f"\nLoading {tool_state} data from: {folder}")
    print(f"Found TDMS files: {len(tdms_files)}")

    for tdms_path in tdms_files:
        print(f"\nReading: {tdms_path}")

        strokes, metadata, n_edges = extract_strokes_from_tdms(tdms_path, tool_state)

        print(f"Trigger rising edges: {n_edges}")
        print(f"Valid strokes extracted: {strokes.shape[0]}")
        print(f"Metadata rows: {len(metadata)}")

        all_strokes.append(strokes)
        all_metadata.append(metadata)

    all_strokes = np.concatenate(all_strokes, axis=0)
    all_metadata = pd.concat(all_metadata, ignore_index=True)

    # Assign global IDs only to valid strokes, matching all_strokes order
    valid_indices = all_metadata.index[all_metadata["valid"] == True].tolist()

    for i, idx in enumerate(valid_indices):
        all_metadata.loc[idx, "global_sample_id"] = f"{tool_state}_{i:06d}"

    valid_metadata = all_metadata[all_metadata["valid"] == True].copy().reset_index(drop=True)

    return all_strokes, valid_metadata, all_metadata


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(strokes, metadata):
    rows = []

    for i, stroke in enumerate(strokes):
        row = {
            "global_sample_id": metadata.loc[i, "global_sample_id"],
            "tool_state": metadata.loc[i, "tool_state"],
            "label": metadata.loc[i, "label"],
            "source_file": metadata.loc[i, "source_file"],
            "local_stroke_id": metadata.loc[i, "local_stroke_id"],
            "raw_start": metadata.loc[i, "raw_start"],
            "raw_end": metadata.loc[i, "raw_end"],
            "raw_length": metadata.loc[i, "raw_length"],
        }

        for c, ch in enumerate(FORCE_CHANNELS):
            x = stroke[:, c]

            row[f"{ch}_max"] = np.max(x)
            row[f"{ch}_min"] = np.min(x)
            row[f"{ch}_mean"] = np.mean(x)
            row[f"{ch}_std"] = np.std(x)
            row[f"{ch}_range"] = np.max(x) - np.min(x)
            row[f"{ch}_rms"] = np.sqrt(np.mean(x**2))
            row[f"{ch}_area"] = safe_area(x)
            row[f"{ch}_abs_area"] = safe_area(np.abs(x))
            row[f"{ch}_peak_index"] = np.argmax(x)
            row[f"{ch}_skew"] = skew(x)
            row[f"{ch}_kurtosis"] = kurtosis(x)

            peak_idx = int(np.argmax(x))

            if peak_idx > 5:
                row[f"{ch}_loading_slope"] = (x[peak_idx] - x[0]) / peak_idx
            else:
                row[f"{ch}_loading_slope"] = 0.0

            if peak_idx < len(x) - 5:
                row[f"{ch}_unloading_slope"] = (x[-1] - x[peak_idx]) / (len(x) - peak_idx)
            else:
                row[f"{ch}_unloading_slope"] = 0.0

            # Settling features
            peak_value = x[peak_idx]
            threshold = 0.05 * abs(peak_value)

            after_peak = np.abs(x[peak_idx:])
            below = np.where(after_peak < threshold)[0]

            if len(below) > 0:
                row[f"{ch}_settling_index_after_peak"] = int(below[0])
            else:
                row[f"{ch}_settling_index_after_peak"] = len(x) - peak_idx

            # FFT features
            x_centered = x - np.mean(x)
            fft_vals = np.abs(np.fft.rfft(x_centered))

            row[f"{ch}_fft_energy_total"] = np.sum(fft_vals**2)
            row[f"{ch}_fft_energy_low"] = np.sum(fft_vals[1:20]**2)
            row[f"{ch}_fft_energy_mid"] = np.sum(fft_vals[20:80]**2)
            row[f"{ch}_fft_energy_high"] = np.sum(fft_vals[80:]**2)

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# MODELS
# ============================================================

def run_pca(X_train, X_all):
    model = PCA(n_components=0.95)
    model.fit(X_train)

    X_rec = model.inverse_transform(model.transform(X_all))
    scores = np.mean((X_all - X_rec) ** 2, axis=1)

    return model, scores


def run_isolation_forest(X_train, X_all):
    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42
    )
    model.fit(X_train)

    scores = -model.score_samples(X_all)

    return model, scores


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec


def run_autoencoder(X_train, X_all, epochs=120, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(X_train_tensor),
        batch_size=batch_size,
        shuffle=True
    )

    model = Autoencoder(input_dim=X_train.shape[1], latent_dim=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()

    for epoch in range(epochs):
        epoch_losses = []

        for batch in train_loader:
            x = batch[0].to(device)

            optimizer.zero_grad()
            x_rec = model(x)
            loss = loss_fn(x_rec, x)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        if (epoch + 1) % 20 == 0:
            print(f"AE epoch {epoch + 1}/{epochs}, loss={np.mean(epoch_losses):.6f}")

    model.eval()

    with torch.no_grad():
        X_all_t = torch.tensor(X_all, dtype=torch.float32).to(device)
        X_all_rec = model(X_all_t).cpu().numpy()

    scores = np.mean((X_all - X_all_rec) ** 2, axis=1)

    return model, scores


# ============================================================
# PLOTS
# ============================================================

def plot_score_histograms(results_df, output_path):
    methods = ["pca_score", "isolation_forest_score", "autoencoder_score"]

    for method in methods:
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

        fname = os.path.join(output_path, f"{method}_new_vs_wear_hist.png")
        plt.savefig(fname, dpi=200)
        plt.close()


def plot_boxplots(results_df, output_path):
    methods = ["pca_score", "isolation_forest_score", "autoencoder_score"]

    for method in methods:
        plt.figure(figsize=(7, 6))

        data = [
            results_df.loc[results_df["tool_state"] == "new", method].values,
            results_df.loc[results_df["tool_state"] == "wear", method].values,
        ]

        plt.boxplot(data, labels=["new", "wear"])
        plt.ylabel("Anomaly score")
        plt.title(f"{method}: score comparison")
        plt.tight_layout()

        fname = os.path.join(output_path, f"{method}_boxplot.png")
        plt.savefig(fname, dpi=200)
        plt.close()


def plot_mean_strokes(strokes_new, strokes_wear, output_path):
    for c, ch in enumerate(FORCE_CHANNELS):
        plt.figure(figsize=(12, 6))

        mean_new = strokes_new[:, :, c].mean(axis=0)
        std_new = strokes_new[:, :, c].std(axis=0)

        mean_wear = strokes_wear[:, :, c].mean(axis=0)
        std_wear = strokes_wear[:, :, c].std(axis=0)

        x = np.arange(RESAMPLED_LENGTH)

        plt.plot(x, mean_new, label="new mean")
        plt.fill_between(x, mean_new - std_new, mean_new + std_new, alpha=0.2)

        plt.plot(x, mean_wear, label="wear mean")
        plt.fill_between(x, mean_wear - std_wear, mean_wear + std_wear, alpha=0.2)

        plt.xlabel("Resampled time index")
        plt.ylabel(ch)
        plt.title(f"Mean ± std stroke: {ch}")
        plt.legend()
        plt.tight_layout()

        fname = os.path.join(output_path, f"mean_std_{ch}.png")
        plt.savefig(fname, dpi=200)
        plt.close()


# ============================================================
# EVALUATION
# ============================================================

def evaluate_scores(results_df):
    y_true = results_df["label"].values.astype(int)

    rows = []

    for method in ["pca_score", "isolation_forest_score", "autoencoder_score"]:
        scores = results_df[method].values

        roc_auc = roc_auc_score(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)

        new_mean = results_df.loc[results_df["tool_state"] == "new", method].mean()
        wear_mean = results_df.loc[results_df["tool_state"] == "wear", method].mean()

        rows.append({
            "method": method,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "new_mean_score": new_mean,
            "wear_mean_score": wear_mean,
            "wear_minus_new_mean": wear_mean - new_mean,
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_dir", default="new", help="Folder containing new-condition TDMS files")
    parser.add_argument("--wear_dir", default="wear", help="Folder containing wear-condition TDMS files")
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    print("=" * 80)
    print("FORCE-ONLY NEW VS WEAR BASELINE")
    print("=" * 80)
    print("Using Version B force channels:")
    for ch in FORCE_CHANNELS:
        print(f"  - {ch}")

    # -----------------------------
    # Load data
    # -----------------------------
    strokes_new, valid_meta_new, all_meta_new = load_dataset(args.new_dir, "new")
    strokes_wear, valid_meta_wear, all_meta_wear = load_dataset(args.wear_dir, "wear")

    print("\nFinal stroke tensors:")
    print("new:", strokes_new.shape)
    print("wear:", strokes_wear.shape)

    # Save raw quality metadata
    all_meta_new.to_csv(os.path.join(args.output_dir, "all_cycle_quality_new.csv"), index=False)
    all_meta_wear.to_csv(os.path.join(args.output_dir, "all_cycle_quality_wear.csv"), index=False)

    valid_meta_new.to_csv(os.path.join(args.output_dir, "valid_cycle_metadata_new.csv"), index=False)
    valid_meta_wear.to_csv(os.path.join(args.output_dir, "valid_cycle_metadata_wear.csv"), index=False)

    np.save(os.path.join(args.output_dir, "strokes_new_resampled.npy"), strokes_new)
    np.save(os.path.join(args.output_dir, "strokes_wear_resampled.npy"), strokes_wear)

    plot_mean_strokes(strokes_new, strokes_wear, args.output_dir)

    # -----------------------------
    # Features
    # -----------------------------
    features_new = extract_features(strokes_new, valid_meta_new)
    features_wear = extract_features(strokes_wear, valid_meta_wear)

    features_all = pd.concat([features_new, features_wear], ignore_index=True)

    features_new.to_csv(os.path.join(args.output_dir, "force_features_new.csv"), index=False)
    features_wear.to_csv(os.path.join(args.output_dir, "force_features_wear.csv"), index=False)
    features_all.to_csv(os.path.join(args.output_dir, "force_features_all.csv"), index=False)

    id_cols = [
        "global_sample_id",
        "tool_state",
        "label",
        "source_file",
        "local_stroke_id",
        "raw_start",
        "raw_end",
        "raw_length",
    ]

    feature_cols = [c for c in features_all.columns if c not in id_cols]

    X_new = features_new[feature_cols].values
    X_all = features_all[feature_cols].values

    # Replace possible numerical issues
    X_new = np.nan_to_num(X_new, nan=0.0, posinf=0.0, neginf=0.0)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_new_scaled = scaler.fit_transform(X_new)
    X_all_scaled = scaler.transform(X_all)

    joblib.dump(scaler, os.path.join(args.output_dir, "feature_scaler.joblib"))
    pd.Series(feature_cols).to_csv(os.path.join(args.output_dir, "feature_columns.csv"), index=False)

    # -----------------------------
    # Train on new, score all
    # -----------------------------
    print("\nTraining PCA on new and scoring new + wear...")
    pca_model, pca_scores = run_pca(X_new_scaled, X_all_scaled)

    print("Training Isolation Forest on new and scoring new + wear...")
    if_model, if_scores = run_isolation_forest(X_new_scaled, X_all_scaled)

    print("Training Autoencoder on new and scoring new + wear...")
    ae_model, ae_scores = run_autoencoder(X_new_scaled, X_all_scaled)

    # -----------------------------
    # Results
    # -----------------------------
    results_df = features_all[id_cols].copy()
    results_df["pca_score"] = pca_scores
    results_df["isolation_forest_score"] = if_scores
    results_df["autoencoder_score"] = ae_scores

    results_df.to_csv(os.path.join(args.output_dir, "model_scores_new_vs_wear.csv"), index=False)

    eval_df = evaluate_scores(results_df)
    eval_df.to_csv(os.path.join(args.output_dir, "model_comparison_metrics.csv"), index=False)

    print("\nModel comparison:")
    print(eval_df)

    plot_score_histograms(results_df, args.output_dir)
    plot_boxplots(results_df, args.output_dir)

    # Save models
    joblib.dump(pca_model, os.path.join(args.output_dir, "pca_model.joblib"))
    joblib.dump(if_model, os.path.join(args.output_dir, "isolation_forest_model.joblib"))
    torch.save(ae_model.state_dict(), os.path.join(args.output_dir, "autoencoder_model.pt"))

    # Top anomalies
    for method in ["pca_score", "isolation_forest_score", "autoencoder_score"]:
        top = results_df.sort_values(method, ascending=False).head(30)
        top.to_csv(os.path.join(args.output_dir, f"top_30_anomalies_{method}.csv"), index=False)

    print("\nDone.")
    print(f"Outputs saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
