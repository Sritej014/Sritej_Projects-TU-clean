from nptdms import TdmsFile
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
import glob

tdms_path = sorted(glob.glob("*.tdms"))[0]
tdms = TdmsFile.read(tdms_path)
group = tdms.groups()[0]

force_channel = "Modul7_Oberwerkzeug"
trigger_channel = "Trigger"

force = np.asarray(group[force_channel][:])
trigger = np.asarray(group[trigger_channel][:])

# Find trigger rising edges
threshold = 0.5 * (np.nanmax(trigger) + np.nanmin(trigger))
binary = trigger > threshold
rising_edges = np.where(np.diff(binary.astype(int)) == 1)[0]

print("Number of rising edges:", len(rising_edges))
print("First 10 rising edges:", rising_edges[:10])

# Choose one stroke
stroke_id = 10
start = rising_edges[stroke_id]
end = rising_edges[stroke_id + 1]

print(f"Stroke {stroke_id}: start={start}, end={end}, length={end-start}")

# Plot raw region around stroke
margin = 1000
raw_start = max(0, start - margin)
raw_end = min(len(force), end + margin)

plt.figure(figsize=(14, 6))
plt.plot(np.arange(raw_start, raw_end), force[raw_start:raw_end], label=force_channel)
plt.plot(np.arange(raw_start, raw_end), trigger[raw_start:raw_end], label="Trigger", alpha=0.7)
plt.axvline(start, color="black", linestyle="--", label="stroke start")
plt.axvline(end, color="red", linestyle="--", label="stroke end")
plt.xlabel("Raw sample index")
plt.ylabel("Signal")
plt.title(f"Origin of extracted stroke {stroke_id} in raw TDMS data")
plt.legend()
plt.tight_layout()
plt.savefig("stroke_origin_raw.png", dpi=200)
plt.close()

# Plot extracted and resampled stroke
raw_stroke = force[start:end]
resampled_stroke = resample(raw_stroke, 512)

plt.figure(figsize=(12, 5))
plt.plot(raw_stroke, label="Raw stroke")
plt.xlabel("Raw stroke sample index")
plt.ylabel(force_channel)
plt.title(f"Raw extracted stroke {stroke_id}")
plt.legend()
plt.tight_layout()
plt.savefig("stroke_raw_extracted.png", dpi=200)
plt.close()

plt.figure(figsize=(12, 5))
plt.plot(resampled_stroke, label="Resampled stroke")
plt.xlabel("Resampled index")
plt.ylabel(force_channel)
plt.title(f"Resampled stroke {stroke_id}")
plt.legend()
plt.tight_layout()
plt.savefig("stroke_resampled_extracted.png", dpi=200)
plt.close()

print("Saved:")
print("stroke_origin_raw.png")
print("stroke_raw_extracted.png")
print("stroke_resampled_extracted.png")
