
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

strokes = np.load("force_baseline_outputs/strokes_resampled.npy")
scores = pd.read_csv("force_baseline_outputs/baseline_scores_new_test.csv")

print(scores.head())
print(scores.describe())

# This file only contains test split scores, not original stroke ids.
# For now, inspect score distribution only.
# In the next version we will preserve indices for exact stroke tracing.

for col in scores.columns:
    top_idx = np.argsort(scores[col].values)[-10:][::-1]
    print(f"\nTop 10 test anomalies by {col}:")
    print(scores.iloc[top_idx][[col]])
