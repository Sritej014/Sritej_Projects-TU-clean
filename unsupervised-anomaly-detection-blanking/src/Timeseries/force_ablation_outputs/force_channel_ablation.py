import os
import subprocess
import pandas as pd


SCRIPT = "force_train_score_compare.py"

EXPERIMENTS = {
    "A1_Modul7_Oberwerkzeug_only": [
        "Modul7_Oberwerkzeug"
    ],
    "A2_Modul10_Oberwerkzeug_only": [
        "Modul10_Oberwerkzeug"
    ],
    "A3_Modul7_Unterwerkzeug_only": [
        "Modul7_Unterwerkzeug"
    ],
    "A4_Modul10_Unterwerkzeug_only": [
        "Modul10_Unterwerkzeug"
    ],
    "A5_lower_pair": [
        "Modul7_Unterwerkzeug",
        "Modul10_Unterwerkzeug"
    ],
    "A6_upper7_plus_lower7": [
        "Modul7_Oberwerkzeug",
        "Modul7_Unterwerkzeug"
    ],
    "A7_versionB_upper7_lower7_lower10": [
        "Modul7_Oberwerkzeug",
        "Modul7_Unterwerkzeug",
        "Modul10_Unterwerkzeug"
    ],
    "A8_all4_channels": [
        "Modul7_Oberwerkzeug",
        "Modul10_Oberwerkzeug",
        "Modul7_Unterwerkzeug",
        "Modul10_Unterwerkzeug"
    ],
}


def patch_force_channels(channels):
    """
    Rewrites FORCE_CHANNELS in force_train_score_compare.py.
    This assumes FORCE_CHANNELS is defined near the top as a Python list.
    """
    with open(SCRIPT, "r") as f:
        text = f.read()

    start = text.index("FORCE_CHANNELS = [")
    end = text.index("]", start) + 1

    new_block = "FORCE_CHANNELS = [\n"
    for ch in channels:
        new_block += f'    "{ch}",\n'
    new_block += "]"

    patched = text[:start] + new_block + text[end:]

    with open(SCRIPT, "w") as f:
        f.write(patched)


def run_experiment(name, channels):
    output_dir = f"force_ablation_outputs/{name}"

    print("\n" + "=" * 100)
    print(f"Running {name}")
    print("Channels:")
    for ch in channels:
        print(f"  - {ch}")
    print("=" * 100)

    patch_force_channels(channels)

    cmd = [
        "python",
        SCRIPT,
        "--new_dir", "new",
        "--wear_dir", "wear",
        "--output_dir", output_dir
    ]

    subprocess.run(cmd, check=True)

    metrics_path = os.path.join(output_dir, "model_comparison_metrics.csv")
    metrics = pd.read_csv(metrics_path)
    metrics["experiment"] = name
    metrics["channels"] = " + ".join(channels)

    return metrics


def main():
    os.makedirs("force_ablation_outputs", exist_ok=True)

    all_metrics = []

    for name, channels in EXPERIMENTS.items():
        metrics = run_experiment(name, channels)
        all_metrics.append(metrics)

    summary = pd.concat(all_metrics, ignore_index=True)

    cols = [
        "experiment",
        "channels",
        "method",
        "roc_auc",
        "pr_auc",
        "new_mean_score",
        "wear_mean_score",
        "wear_minus_new_mean",
    ]

    summary = summary[cols]
    summary.to_csv("force_ablation_outputs/channel_ablation_summary.csv", index=False)

    print("\n" + "=" * 100)
    print("CHANNEL ABLATION SUMMARY")
    print("=" * 100)
    print(summary.to_string(index=False))
    print("\nSaved: force_ablation_outputs/channel_ablation_summary.csv")


if __name__ == "__main__":
    main()
