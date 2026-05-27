from nptdms import TdmsFile
import glob
import numpy as np
import matplotlib.pyplot as plt

tdms_path = sorted(glob.glob("*.tdms"))[0]
tdms = TdmsFile.read(tdms_path)
group = tdms.groups()[0]

channels = [
    "Trigger",
    "DI1",
    "DI2",
    "Modul7_Oberwerkzeug",
    "Modul10_Oberwerkzeug",
    "Modul7_Unterwerkzeug",
    "Modul10_Unterwerkzeug",
    "leer",
    "Ticks",
    "Timestamp",
]

for ch in channels:
    if ch not in group:
        print(f"Missing: {ch}")
        continue

    x = group[ch][:]
    x = np.asarray(x)

    plt.figure(figsize=(14, 4))
    plt.plot(x)
    plt.title(ch)
    plt.xlabel("Sample index")
    plt.ylabel(ch)
    plt.tight_layout()
    fname = f"channel_{ch}.png".replace("/", "_")
    plt.savefig(fname, dpi=200)
    plt.close()

    print(f"Saved {fname}")
