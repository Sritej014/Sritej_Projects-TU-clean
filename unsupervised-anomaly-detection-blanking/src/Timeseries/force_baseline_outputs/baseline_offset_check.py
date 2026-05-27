import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT = "force_new_vs_wear_outputs"
ROBUST_OUT = os.path.join(OUT, "robustness_checks")
os.makedirs(ROBUST_OUT, exist_ok=True)

CHANNELS = [
    "Modul7_Oberwerkzeug",
    "Modul7_Unterwerkzeug",
    "Modul10_Unterwerkzeug",
]

PRECONTACT_END = 80


def main():
    strokes_new = np.load(os.path.join(OUT, "strokes_new_resampled.npy"))
    strokes_wear = np.load(os.path.join(OUT, "strokes_wear_resampled.npy"))

    rows = []

    for state, strokes in [("new", strokes_new), ("wear", strokes_wear)]:
        for i in range(strokes.shape[0]):
            row = {
                "tool_state": state,
                "stroke_index": i,
            }

            for c, ch in enumerate(CHANNELS):
                x = strokes[i, :, c]

                pre = x[:PRECONTACT_END]
                row[f"{ch}_precontact_mean"] = np.mean(pre)
                row[f"{ch}_precontact_std"] = np.std(pre)
                row[f"{ch}_precontact_min"] = np.min(pre)
                row[f"{ch}_precontact_max"] = np.max(pre)

                # Also check post-contact negative dip for upper tool
                row[f"{ch}_post_250_380_min"] = np.min(x[250:380])
                row[f"{ch}_post_250_380_mean"] = np.mean(x[250:380])

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROBUST_OUT, "baseline_offset_per_stroke.csv"), index=False)

    print("\n=== Baseline offset summary ===")
    for ch in CHANNELS:
        col = f"{ch}_precontact_mean"
        print("\n", col)
        print(df.groupby("tool_state")[col].describe())

    # Plot each channel baseline offset
    for ch in CHANNELS:
        col = f"{ch}_precontact_mean"

        plt.figure(figsize=(10, 6))
        plt.hist(df.loc[df["tool_state"] == "new", col], bins=40, alpha=0.6, label="new")
        plt.hist(df.loc[df["tool_state"] == "wear", col], bins=40, alpha=0.6, label="wear")
        plt.xlabel("Pre-contact baseline mean")
        plt.ylabel("Count")
        plt.title(f"Pre-contact baseline offset: {ch}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(ROBUST_OUT, f"baseline_offset_{ch}.png"), dpi=200)
        plt.close()

    # Upper tool post-contact negative behavior
    ch = "Modul7_Oberwerkzeug"
    col = f"{ch}_post_250_380_min"

    plt.figure(figsize=(10, 6))
    plt.hist(df.loc[df["tool_state"] == "new", col], bins=40, alpha=0.6, label="new")
    plt.hist(df.loc[df["tool_state"] == "wear", col], bins=40, alpha=0.6, label="wear")
    plt.xlabel("Minimum value in post-contact window 250:380")
    plt.ylabel("Count")
    plt.title("Post-contact negative dip: Modul7_Oberwerkzeug")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ROBUST_OUT, "post_contact_negative_dip_Modul7_Oberwerkzeug.png"), dpi=200)
    plt.close()

    print(f"\nSaved baseline offset outputs in: {ROBUST_OUT}")


if __name__ == "__main__":
    main()
