import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nptdms import TdmsFile
from scipy.signal import find_peaks, resample
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# CONFIG
# -----------------------------

FORCE_CHANNELS = [
    "Modul7_Oberwerkzeug",
    "Modul10_Oberwerkzeug",
    "Modul7_Unterwerkzeug",
    "Modul10_Unterwerkzeug",
]

TRIGGER_CHANNEL = "Trigger"
TIMESTAMP_CHANNEL = "Timestamp"

RESAMPLED_LENGTH = 512
MIN_STROKE_LENGTH = 100
PEAK_DISTANCE = 300
PEAK_PROMINENCE = None

OUTPUT_DIR = "force_baseline_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# TDMS READING
# -----------------------------

def read_tdms_file(tdms_path):
    tdms_file = TdmsFile.read(tdms_path)

    groups = tdms_file.groups()
    if len(groups) == 0:
        raise ValueError("No TDMS groups found.")

    group = groups[0]
    group_name = group.name

    data = {}

    for ch in FORCE_CHANNELS + [TRIGGER_CHANNEL, TIMESTAMP_CHANNEL]:
        if ch in group:
            data[ch] = group[ch][:]
        else:
            print(f"Warning: channel {ch} not found in {tdms_path}")

    return group_name, data


# -----------------------------
# STROKE SEGMENTATION
# -----------------------------

def segment_by_trigger(trigger, min_length=MIN_STROKE_LENGTH):
    """
    Segment using rising edges in Trigger.
    Works if trigger is digital or step-like.
    """
    trigger = np.asarray(trigger)
    threshold = 0.5 * (np.nanmax(trigger) + np.nanmin(trigger))
    binary = trigger > threshold

    rising_edges = np.where(np.diff(binary.astype(int)) == 1)[0]

    segments = []
    for i in range(len(rising_edges) - 1):
        start = rising_edges[i]
        end = rising_edges[i + 1]
        if end - start >= min_length:
            segments.append((start, end))

    return segments


def segment_by_force_peaks(force_signal, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE):
    """
    Fallback segmentation using force peaks.
    Uses one force channel or combined force magnitude.
    """
    x = np.asarray(force_signal)
    x = np.nan_to_num(x)

    if prominence is None:
        prominence = 0.2 * np.std(x)

    peaks, _ = find_peaks(x, distance=distance, prominence=prominence)

    segments = []
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]
        if end - start >= MIN_STROKE_LENGTH:
            segments.append((start, end))

    return segments


def extract_strokes_from_data(data):
    """
    Returns array of shape:
    n_strokes x RESAMPLED_LENGTH x n_channels
    """

    force_matrix = np.vstack([data[ch] for ch in FORCE_CHANNELS]).T

    if TRIGGER_CHANNEL in data:
        segments = segment_by_trigger(data[TRIGGER_CHANNEL])
        print(f"Segments found using trigger: {len(segments)}")
    else:
        combined_force = np.linalg.norm(force_matrix, axis=1)
        segments = segment_by_force_peaks(combined_force)
        print(f"Segments found using force peaks: {len(segments)}")

    if len(segments) < 2:
        print("Trigger segmentation weak. Trying force peak segmentation.")
        combined_force = np.linalg.norm(force_matrix, axis=1)
        segments = segment_by_force_peaks(combined_force)
        print(f"Segments found using force peaks: {len(segments)}")

    strokes = []

    for start, end in segments:
        stroke = force_matrix[start:end, :]

        if stroke.shape[0] < MIN_STROKE_LENGTH:
            continue

        stroke_resampled = resample(stroke, RESAMPLED_LENGTH, axis=0)
        strokes.append(stroke_resampled)

    if len(strokes) == 0:
        raise ValueError("No valid strokes extracted.")

    return np.stack(strokes, axis=0), segments


# -----------------------------
# FEATURE EXTRACTION
# -----------------------------

def extract_features(strokes):
    """
    strokes shape:
    n_strokes x time x channels
    """

    rows = []

    for i, stroke in enumerate(strokes):
        row = {"stroke_id": i}

        for c, ch_name in enumerate(FORCE_CHANNELS):
            x = stroke[:, c]

            row[f"{ch_name}_max"] = np.max(x)
            row[f"{ch_name}_min"] = np.min(x)
            row[f"{ch_name}_mean"] = np.mean(x)
            row[f"{ch_name}_std"] = np.std(x)
            row[f"{ch_name}_rms"] = np.sqrt(np.mean(x**2))
            row[f"{ch_name}_area"] = np.trapezoid(x)
            row[f"{ch_name}_peak_index"] = np.argmax(x)
            row[f"{ch_name}_skew"] = skew(x)
            row[f"{ch_name}_kurtosis"] = kurtosis(x)

            # Approximate loading/unloading slopes
            peak_idx = np.argmax(x)
            if peak_idx > 5:
                row[f"{ch_name}_loading_slope"] = (x[peak_idx] - x[0]) / peak_idx
            else:
                row[f"{ch_name}_loading_slope"] = 0.0

            if peak_idx < len(x) - 5:
                row[f"{ch_name}_unloading_slope"] = (x[-1] - x[peak_idx]) / (len(x) - peak_idx)
            else:
                row[f"{ch_name}_unloading_slope"] = 0.0

            # FFT features
            fft_vals = np.abs(np.fft.rfft(x - np.mean(x)))
            row[f"{ch_name}_fft_energy_total"] = np.sum(fft_vals**2)
            row[f"{ch_name}_fft_energy_low"] = np.sum(fft_vals[1:20]**2)
            row[f"{ch_name}_fft_energy_mid"] = np.sum(fft_vals[20:80]**2)
            row[f"{ch_name}_fft_energy_high"] = np.sum(fft_vals[80:]**2)

        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------
# PCA ANOMALY MODEL
# -----------------------------

def run_pca_anomaly(X_train, X_test, n_components=0.95):
    pca = PCA(n_components=n_components)
    pca.fit(X_train)

    X_train_rec = pca.inverse_transform(pca.transform(X_train))
    X_test_rec = pca.inverse_transform(pca.transform(X_test))

    train_scores = np.mean((X_train - X_train_rec) ** 2, axis=1)
    test_scores = np.mean((X_test - X_test_rec) ** 2, axis=1)

    return pca, train_scores, test_scores


# -----------------------------
# ISOLATION FOREST
# -----------------------------

def run_isolation_forest(X_train, X_test):
    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42
    )
    model.fit(X_train)

    # sklearn gives higher score for normal.
    # We invert so higher = more anomalous.
    train_scores = -model.score_samples(X_train)
    test_scores = -model.score_samples(X_test)

    return model, train_scores, test_scores


# -----------------------------
# AUTOENCODER
# -----------------------------

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


def run_autoencoder(X_train, X_test, epochs=100, batch_size=32, lr=1e-3):
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
        losses = []

        for batch in train_loader:
            x = batch[0].to(device)

            optimizer.zero_grad()
            x_rec = model(x)
            loss = loss_fn(x_rec, x)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if (epoch + 1) % 20 == 0:
            print(f"AE epoch {epoch + 1}/{epochs}, loss={np.mean(losses):.6f}")

    model.eval()

    with torch.no_grad():
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

        train_rec = model(X_train_t).cpu().numpy()
        test_rec = model(X_test_t).cpu().numpy()

    train_scores = np.mean((X_train - train_rec) ** 2, axis=1)
    test_scores = np.mean((X_test - test_rec) ** 2, axis=1)

    return model, train_scores, test_scores


# -----------------------------
# PLOTTING
# -----------------------------

def plot_scores(scores_dict, output_path):
    plt.figure(figsize=(10, 6))

    for name, scores in scores_dict.items():
        plt.hist(scores, bins=40, alpha=0.5, label=name)

    plt.xlabel("Anomaly score")
    plt.ylabel("Count")
    plt.title("Anomaly score distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_strokes(strokes, output_path, n=20):
    plt.figure(figsize=(12, 7))

    n_plot = min(n, strokes.shape[0])

    for i in range(n_plot):
        plt.plot(strokes[i, :, 0], alpha=0.4)

    plt.xlabel("Resampled time index")
    plt.ylabel(FORCE_CHANNELS[0])
    plt.title(f"First {n_plot} resampled strokes: {FORCE_CHANNELS[0]}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# -----------------------------
# MAIN
# -----------------------------

def main():
    tdms_files = sorted(glob.glob("*.tdms"))

    if len(tdms_files) == 0:
        raise FileNotFoundError("No .tdms files found in current directory.")

    all_strokes = []
    metadata_rows = []

    for tdms_path in tdms_files:
        print(f"\nReading: {tdms_path}")
        group_name, data = read_tdms_file(tdms_path)

        strokes, segments = extract_strokes_from_data(data)
        print(f"Extracted strokes shape: {strokes.shape}")

        for i in range(strokes.shape[0]):
            metadata_rows.append({
                "file": tdms_path,
                "group": group_name,
                "local_stroke_id": i,
                "start_index": segments[i][0] if i < len(segments) else None,
                "end_index": segments[i][1] if i < len(segments) else None,
                "tool_state": "new"
            })

        all_strokes.append(strokes)

    all_strokes = np.concatenate(all_strokes, axis=0)
    metadata = pd.DataFrame(metadata_rows)

    print("\nTotal strokes:", all_strokes.shape)

    np.save(os.path.join(OUTPUT_DIR, "strokes_resampled.npy"), all_strokes)
    metadata.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)

    plot_strokes(
        all_strokes,
        os.path.join(OUTPUT_DIR, "resampled_strokes_preview.png")
    )

    features = extract_features(all_strokes)
    features.to_csv(os.path.join(OUTPUT_DIR, "force_features.csv"), index=False)

    feature_cols = [c for c in features.columns if c != "stroke_id"]
    X = features[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))

    # Since this folder is only "new", train/test split is normal-only.
    X_train, X_test = train_test_split(
        X_scaled,
        test_size=0.25,
        random_state=42
    )

    print("\nRunning PCA anomaly baseline...")
    pca_model, pca_train_scores, pca_test_scores = run_pca_anomaly(X_train, X_test)

    print("Running Isolation Forest baseline...")
    if_model, if_train_scores, if_test_scores = run_isolation_forest(X_train, X_test)

    print("Running Autoencoder baseline...")
    ae_model, ae_train_scores, ae_test_scores = run_autoencoder(X_train, X_test)

    scores_df = pd.DataFrame({
        "pca_score": pca_test_scores,
        "isolation_forest_score": if_test_scores,
        "autoencoder_score": ae_test_scores,
    })

    scores_df.to_csv(os.path.join(OUTPUT_DIR, "baseline_scores_new_test.csv"), index=False)

    plot_scores(
        {
            "PCA": pca_test_scores,
            "Isolation Forest": if_test_scores,
            "Autoencoder": ae_test_scores,
        },
        os.path.join(OUTPUT_DIR, "anomaly_score_histograms_new_only.png")
    )

    joblib.dump(pca_model, os.path.join(OUTPUT_DIR, "pca_model.joblib"))
    joblib.dump(if_model, os.path.join(OUTPUT_DIR, "isolation_forest_model.joblib"))
    torch.save(ae_model.state_dict(), os.path.join(OUTPUT_DIR, "autoencoder_model.pt"))

    print("\nDone.")
    print(f"Outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()