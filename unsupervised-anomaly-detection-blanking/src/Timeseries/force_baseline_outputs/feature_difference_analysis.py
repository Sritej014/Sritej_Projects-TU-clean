import os
import numpy as np
import pandas as pd


OUT = "force_new_vs_wear_outputs_baseline_corrected"
if not os.path.exists(OUT):
    OUT = "force_new_vs_wear_outputs"

df = pd.read_csv(os.path.join(OUT, "force_features_all.csv"))

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

feature_cols = [c for c in df.columns if c not in id_cols]

rows = []

for col in feature_cols:
    new_vals = df.loc[df["tool_state"] == "new", col].values
    wear_vals = df.loc[df["tool_state"] == "wear", col].values

    new_mean = np.nanmean(new_vals)
    wear_mean = np.nanmean(wear_vals)

    new_std = np.nanstd(new_vals)
    wear_std = np.nanstd(wear_vals)
    pooled_std = np.sqrt(0.5 * (new_std**2 + wear_std**2))

    effect = (wear_mean - new_mean) / pooled_std if pooled_std > 1e-12 else np.nan

    rows.append({
        "feature": col,
        "new_mean": new_mean,
        "wear_mean": wear_mean,
        "wear_minus_new": wear_mean - new_mean,
        "new_std": new_std,
        "wear_std": wear_std,
        "standardized_effect_size": effect,
        "abs_effect_size": abs(effect) if np.isfinite(effect) else np.nan,
    })

out = pd.DataFrame(rows).sort_values("abs_effect_size", ascending=False)
out.to_csv(os.path.join(OUT, "feature_difference_ranking.csv"), index=False)

print("\nTop 40 separating features:")
print(out.head(40).to_string(index=False))
print(f"\nSaved: {os.path.join(OUT, 'feature_difference_ranking.csv')}")
