import numpy as np
import matplotlib.pyplot as plt

strokes = np.load("force_baseline_outputs/strokes_resampled.npy")

force_channels = [
    "Modul7_Oberwerkzeug",
    "Modul10_Oberwerkzeug",
    "Modul7_Unterwerkzeug",
    "Modul10_Unterwerkzeug",
]

mean_stroke = strokes.mean(axis=0)
std_stroke = strokes.std(axis=0)

plt.figure(figsize=(12, 7))

for i, name in enumerate(force_channels):
    plt.plot(mean_stroke[:, i], label=name)
    plt.fill_between(
        np.arange(mean_stroke.shape[0]),
        mean_stroke[:, i] - std_stroke[:, i],
        mean_stroke[:, i] + std_stroke[:, i],
        alpha=0.15,
    )

plt.xlabel("Resampled time index")
plt.ylabel("Signal value")
plt.title("Mean ± std of four force channels")
plt.legend()
plt.tight_layout()
plt.savefig("force_baseline_outputs/four_force_channels_mean_std.png", dpi=200)
plt.close()

# Correlation between flattened channel signals
X = strokes.reshape(-1, strokes.shape[-1])
corr = np.corrcoef(X.T)

print("Correlation matrix:")
for i, row in enumerate(corr):
    print(force_channels[i], row)

plt.figure(figsize=(6, 5))
plt.imshow(corr, vmin=-1, vmax=1)
plt.xticks(range(4), force_channels, rotation=45, ha="right")
plt.yticks(range(4), force_channels)
plt.colorbar(label="Correlation")
plt.title("Force channel correlation")
plt.tight_layout()
plt.savefig("force_baseline_outputs/force_channel_correlation.png", dpi=200)
plt.close()
