from nptdms import TdmsFile

tdms_file = TdmsFile.read(
    "/home/sritejkumbar/Documents/MT/Data/ZWF/Timeseries/new/24-11-06__18-50-56_SI_neuer_Stempel_500.tdms"
)

for group in tdms_file.groups():
    print("Group:", group.name)

    for channel in group.channels():
        print("  Channel:", channel.name)