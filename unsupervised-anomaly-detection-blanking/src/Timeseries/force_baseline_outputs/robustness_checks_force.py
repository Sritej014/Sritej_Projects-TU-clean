import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT = "force_new_vs_wear_outputs"
ROBUST_OUT = os.path.join(OUT, "robustness_checks")
os.makedirs(ROBUST_OUT, exist_ok=True)


def load_metadata():
    new_meta = pd.read_csv(os.path.join(OUT, "valid_cycle_metadata_new.csv"))
    wear_meta = pd.read_csv(os.path.join(OUT, "valid_cycle_metadata_wear.csv"))

    new_meta["tool_state"] = "new"
    wear_meta["tool_state"] = "wear"

    return pd.concat([new_meta, wear_meta], ignore_index=True)


def compare_raw_lengths(df):
    print("\n=== Raw cycle length comparison ===")
    print(df.groupby("tool_state")["raw_length"].describe())

    summary = df.groupby("tool_state")["raw_length"].agg(
        ["count", "mean", "std", "min", "median", "max"]
    )
    summary.to_csv(os.path.join(ROBUST_OUT, "raw_length_summary.csv"))

    plt.figure(figsize=(10, 6))
    plt.hist(df.loc[df["tool_state"] == "new", "raw_length"], bins=40, alpha=0.6, label="new")
    plt.hist(df.loc[df["tool_state"] == "wear", "raw_length"], bins=40, alpha=0.6, label="wear")
    plt.xlabel("Raw cycle length [samples]")
    plt.ylabel("Count")
    plt.title("Trigger-to-trigger raw cycle length")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ROBUST_OUT, "raw_cycle_length_hist.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.boxplot(
        [
            df.loc[df["tool_state"] == "new", "raw_length"],
            df.loc[df["tool_state"] == "wear", "raw_length"],
        ],
        labels=["new", "wear"],
    )
    plt.ylabel("Raw cycle length [samples]")
    plt.title("Raw cycle length comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(ROBUST_OUT, "raw_cycle_length_boxplot.png"), dpi=200)
    plt.close()


def main():
    df = load_metadata()
    compare_raw_lengths(df)
    print(f"\nSaved robustness outputs in: {ROBUST_OUT}")


if __name__ == "__main__":
    main()
