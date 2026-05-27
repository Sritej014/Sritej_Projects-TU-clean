from nptdms import TdmsFile
import glob
import numpy as np

tdms_files = sorted(glob.glob("*.tdms"))

if not tdms_files:
    raise FileNotFoundError("No TDMS files found.")

tdms_path = tdms_files[0]
print(f"Reading: {tdms_path}")

tdms = TdmsFile.read(tdms_path)

for group in tdms.groups():
    print("\n" + "=" * 80)
    print(f"GROUP: {group.name}")
    print("Group properties:")
    for k, v in group.properties.items():
        print(f"  {k}: {v}")

    print("\nChannels:")
    for ch in group.channels():
        data = ch[:]
        print("\n" + "-" * 60)
        print(f"Channel: {ch.name}")
        print(f"Length: {len(data)}")
        print(f"Dtype: {data.dtype}")

        if len(data) > 0:
            finite = np.asarray(data)
            if np.issubdtype(finite.dtype, np.number):
                print(f"Min: {np.nanmin(finite)}")
                print(f"Max: {np.nanmax(finite)}")
                print(f"Mean: {np.nanmean(finite)}")
                print(f"Std: {np.nanstd(finite)}")
                print(f"First 10 values: {finite[:10]}")
            else:
                print(f"First 10 values: {finite[:10]}")

        print("Properties:")
        for k, v in ch.properties.items():
            print(f"  {k}: {v}")