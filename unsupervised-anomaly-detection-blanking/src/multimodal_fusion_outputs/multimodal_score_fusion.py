import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score


EPS = 1e-12


def zscore_from_new(df, score_col):
    """
    Standardize scores using only the new/normal distribution.
    Higher score = more anomalous.
    """
    new_scores = df.loc[df["tool_state"] == "new", score_col].values.astype(float)
    mu = np.mean(new_scores)
    sigma = np.std(new_scores)

    if sigma < EPS:
        sigma = EPS

    return (df[score_col].values.astype(float) - mu) / sigma, mu, sigma


def load_force_scores(force_csv, force_score_col):
    df = pd.read_csv(force_csv).copy()

    required = ["tool_state", "label", force_score_col]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}' in force CSV: {force_csv}")

    out = pd.DataFrame({
        "tool_state": df["tool_state"].values,
        "label": df["label"].astype(int).values,
        "force_raw_score": df[force_score_col].values.astype(float),
    })

    out["source"] = "force"

    out["force_z"], mu, sigma = zscore_from_new(out, "force_raw_score")

    print("\nForce score normalization:")
    print(f"  score column = {force_score_col}")
    print(f"  new mean = {mu}")
    print(f"  new std  = {sigma}")

    return out


def load_patchcore_scores(csv_path, modality_name):
    df = pd.read_csv(csv_path).copy()

    required = ["tool_state", "label", "patchcore_score"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}' in PatchCore CSV: {csv_path}")

    out = pd.DataFrame({
        "tool_state": df["tool_state"].values,
        "label": df["label"].astype(int).values,
        f"{modality_name}_raw_score": df["patchcore_score"].values.astype(float),
    })

    out["source"] = modality_name
    out[f"{modality_name}_z"], mu, sigma = zscore_from_new(out, f"{modality_name}_raw_score")

    print(f"\n{modality_name} PatchCore score normalization:")
    print(f"  new mean = {mu}")
    print(f"  new std  = {sigma}")

    return out


def split_state(df, state):
    return df[df["tool_state"] == state].reset_index(drop=True)


def build_index_aligned_fusion_table(force_df, punch_df, sheet_df):
    """
    Creates an index-aligned paired subset.
    This is a practical first fusion if exact stroke-image IDs are unavailable.
    """

    paired_rows = []

    for state, label in [("new", 0), ("wear", 1)]:
        f = split_state(force_df, state)
        p = split_state(punch_df, state)
        s = split_state(sheet_df, state)

        n = min(len(f), len(p), len(s))

        print(f"\nPairing state = {state}")
        print(f"  force samples = {len(f)}")
        print(f"  punch samples = {len(p)}")
        print(f"  sheet samples = {len(s)}")
        print(f"  paired samples used = {n}")

        for i in range(n):
            paired_rows.append({
                "paired_id": f"{state}_{i:06d}",
                "tool_state": state,
                "label": label,

                "force_z": f.loc[i, "force_z"],
                "punch_z": p.loc[i, "punch_z"],
                "sheet_z": s.loc[i, "sheet_z"],

                "force_raw_score": f.loc[i, "force_raw_score"],
                "punch_raw_score": p.loc[i, "punch_raw_score"],
                "sheet_raw_score": s.loc[i, "sheet_raw_score"],
            })

    paired = pd.DataFrame(paired_rows)

    return paired


def add_fusion_scores(df):
    """
    Score-level fusion. Since all scores are z-normalized using new data,
    simple averaging is meaningful.
    """

    df = df.copy()

    df["force_only"] = df["force_z"]
    df["punch_patchcore_only"] = df["punch_z"]
    df["sheet_patchcore_only"] = df["sheet_z"]

    df["force_plus_sheet"] = df[["force_z", "sheet_z"]].mean(axis=1)
    df["force_plus_punch"] = df[["force_z", "punch_z"]].mean(axis=1)
    df["force_plus_punch_plus_sheet"] = df[["force_z", "punch_z", "sheet_z"]].mean(axis=1)

    # Optional image-only fusion
    df["punch_plus_sheet"] = df[["punch_z", "sheet_z"]].mean(axis=1)

    return df


def evaluate_scores(df, score_cols):
    y = df["label"].values.astype(int)

    rows = []

    for col in score_cols:
        scores = df[col].values.astype(float)

        rows.append({
            "method": col,
            "roc_auc": roc_auc_score(y, scores),
            "pr_auc": average_precision_score(y, scores),
            "new_mean_score": df.loc[df["tool_state"] == "new", col].mean(),
            "wear_mean_score": df.loc[df["tool_state"] == "wear", col].mean(),
            "wear_minus_new_mean": (
                df.loc[df["tool_state"] == "wear", col].mean()
                - df.loc[df["tool_state"] == "new", col].mean()
            ),
        })

    return pd.DataFrame(rows)


def plot_histograms(df, score_cols, output_dir):
    for col in score_cols:
        plt.figure(figsize=(10, 6))

        plt.hist(
            df.loc[df["tool_state"] == "new", col],
            bins=40,
            alpha=0.6,
            label="new",
        )

        plt.hist(
            df.loc[df["tool_state"] == "wear", col],
            bins=40,
            alpha=0.6,
            label="wear",
        )

        plt.xlabel("Normalized anomaly score")
        plt.ylabel("Count")
        plt.title(f"{col}: new vs wear")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col}_hist.png"), dpi=200)
        plt.close()


def plot_score_scatter(df, output_dir):
    """
    Useful to see whether force/image scores agree.
    """

    pairs = [
        ("force_z", "sheet_z"),
        ("force_z", "punch_z"),
        ("punch_z", "sheet_z"),
    ]

    for xcol, ycol in pairs:
        plt.figure(figsize=(7, 6))

        for state in ["new", "wear"]:
            mask = df["tool_state"] == state
            plt.scatter(
                df.loc[mask, xcol],
                df.loc[mask, ycol],
                s=18,
                alpha=0.7,
                label=state,
            )

        plt.xlabel(xcol)
        plt.ylabel(ycol)
        plt.title(f"{xcol} vs {ycol}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"scatter_{xcol}_vs_{ycol}.png"), dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--force_csv",
        default="force_ablation_outputs/A1_Modul7_Oberwerkzeug_only/model_scores_new_vs_wear.csv",
        help="Force model score CSV. Recommended: upper-tool-only A1 output."
    )

    parser.add_argument(
        "--force_score_col",
        default="pca_score",
        help="Force anomaly score column: pca_score, isolation_forest_score, or autoencoder_score."
    )

    parser.add_argument(
        "--punch_csv",
        default="patchcore_outputs/Punch/patchcore_scores.csv",
        help="Punch PatchCore scores CSV."
    )

    parser.add_argument(
        "--sheet_csv",
        default="patchcore_outputs/Sheet/patchcore_scores.csv",
        help="Sheet PatchCore scores CSV."
    )

    parser.add_argument(
        "--output_dir",
        default="multimodal_fusion_outputs",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 100)
    print("MULTIMODAL SCORE-LEVEL FUSION")
    print("=" * 100)
    print("Important: this script uses index-aligned pairing within each class.")
    print("For true physical fusion, replace this with exact stroke-image ID matching.")
    print("=" * 100)

    force_df = load_force_scores(args.force_csv, args.force_score_col)
    punch_df = load_patchcore_scores(args.punch_csv, "punch")
    sheet_df = load_patchcore_scores(args.sheet_csv, "sheet")

    paired = build_index_aligned_fusion_table(force_df, punch_df, sheet_df)
    paired = add_fusion_scores(paired)

    paired.to_csv(os.path.join(args.output_dir, "paired_multimodal_scores.csv"), index=False)

    score_cols = [
        "force_only",
        "punch_patchcore_only",
        "sheet_patchcore_only",
        "force_plus_sheet",
        "force_plus_punch",
        "force_plus_punch_plus_sheet",
        "punch_plus_sheet",
    ]

    metrics = evaluate_scores(paired, score_cols)
    metrics.to_csv(os.path.join(args.output_dir, "multimodal_fusion_metrics.csv"), index=False)

    print("\nFusion metrics:")
    print(metrics.to_string(index=False))

    plot_histograms(paired, score_cols, args.output_dir)
    plot_score_scatter(paired, args.output_dir)

    print(f"\nSaved outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
