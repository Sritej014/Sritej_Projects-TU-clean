import pandas as pd
import os
from pathlib import Path


def get_ciss_raw(directory):
    """
    This function reads all the .csv files in the directory of the CISS sensor and appends these as dataframes to the list

    Parameters
    ----------
    directory : str
        Path, where the .csv files are

    Returns
    -------
    pd.DataFrame
        Dataframe as concatenation of .csv files
    """
    dataframes = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path, skiprows=8, encoding='Windows-1252')
            df = df.iloc[:-1]  # Remove the last row
            df.rename(columns={"Unix TimeStamp (ms)": "TimeStamp", "Data / x_axis": "x_axis"}, inplace=True)
            df["TimeStamp"] = pd.to_numeric(df["TimeStamp"]).astype(float)
            df.drop(columns=["Measurement Type"], inplace=True)
            if directory.__contains__("Worn"):
                df["Label"] = 1
            if directory.__contains__("Good"):
                df["Label"] = 0
            dataframes.append(df)
    df = pd.concat(dataframes)
    return df


def get_registry_record(directory):
    """
    This function reads all the .csv files in the directory of the registry and appends these as dataframes to the list

    Parameters
    ----------
    directory : str
        Path, where the .csv files are

    Returns
    -------
    pd.DataFrame
        Dataframe as concatenation of .csv files
    """
    dataframes = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            df.rename(columns={"Unnamed: 0": "TimeStamp"}, inplace=True)
            df["TimeStamp"] = df["TimeStamp"].astype(str).str.replace(".", "", regex=False).str[:13]
            df["TimeStamp"] = pd.to_numeric(df["TimeStamp"]).astype(float)
            df.drop(columns=["WinPCNCRunning", "CurrentJobProgress", "CurrentJobTimeMin", "CurrentJobTimeSec", "CurrentJobCommandNo", "CurrentSpdOvr", "State"], inplace=True)
            dataframes.append(df)
    df = pd.concat(dataframes)
    return df


# Define the directories of the .csv files
current_dir = Path(__file__).resolve().parents[2]
"""directories = ["Worn_tool_no1_pattern1", "Worn_tool_no2_pattern2", "Worn_tool_no3_pattern3", 
                        "Good_tool_no1_pattern4", "Good_tool_no2_pattern5"]
"""
directories = ["Worn_tool_no1_pattern1", "Worn_tool_no2_pattern2", 
                        "Good_tool_no1_pattern4"]

directories_val = ["Worn_tool_no3_pattern3", "Good_tool_no2_pattern5"]

# Read out all the csv files for CISS and registry
dataframes_ciss = []
for directory in directories:
    connected_path = os.path.join(current_dir, "data", "raw", directory, "CISSRawData_LogFiles")
    dataframes_ciss.append(get_ciss_raw(connected_path))

dataframes_registry = []
for directory in directories:
    connected_path = os.path.join(current_dir, "data", "raw", directory)
    dataframes_registry.append(get_registry_record(connected_path))

# Validation set
dataframes_ciss_val = []
for directory in directories_val:
    connected_path = os.path.join(current_dir, "data", "raw", directory, "CISSRawData_LogFiles")
    dataframes_ciss_val.append(get_ciss_raw(connected_path))

dataframes_registry_val = []
for directory in directories:
    connected_path = os.path.join(current_dir, "data", "raw", directory)
    dataframes_registry_val.append(get_registry_record(connected_path))

# Concatenate the read dataframes
dataframe_ciss = pd.concat(dataframes_ciss)
dataframe_ciss = dataframe_ciss.sort_values("TimeStamp", ascending=True)
dataframe_registry = pd.concat(dataframes_registry)
dataframe_registry = dataframe_registry.sort_values("TimeStamp", ascending=True)

dataframe_ciss_val = pd.concat(dataframes_ciss_val)
dataframe_ciss_val = dataframe_ciss_val.sort_values("TimeStamp", ascending=True)
dataframe_registry_val = pd.concat(dataframes_registry_val)
dataframe_registry_val = dataframe_registry_val.sort_values("TimeStamp", ascending=True)


dataframe_all_val = pd.merge_asof(dataframe_ciss_val, dataframe_registry_val, on="TimeStamp", direction="nearest")
peak_threshold_val = dataframe_all_val["Pos_Z"].max()
dataframe_all_val = dataframe_all_val[dataframe_all_val["Pos_Z"] >= peak_threshold_val]
dataframe_all_val.drop(columns={"Pos_X", "Pos_Y", "Pos_Z"}, inplace=True)
# Windowize the data
num_windows = 3600     # 36 pockets per tool, 2!!! tools, 50 windows for each pocket
window_size = len(dataframe_all_val) // num_windows
step_size = window_size

windows = []
for i in range(0, len(dataframe_all_val), step_size):
    if i + window_size > len(dataframe_all_val):
        break

    window_data = dataframe_all_val.iloc[i:i+window_size]

    # Feature extraction
    features = {
        'x_mean': window_data['x_axis'].mean(),
        'x_std': window_data['x_axis'].std(),
        'x_sum': window_data['x_axis'].sum(),
        'x_median': window_data['x_axis'].median(),
        'y_mean': window_data['y_axis'].mean(),
        'y_std': window_data['y_axis'].std(),
        'y_sum': window_data['y_axis'].sum(),
        'y_median': window_data['y_axis'].median(),
        'z_mean': window_data['z_axis'].mean(),
        'z_std': window_data['z_axis'].std(),
        'z_sum': window_data['z_axis'].sum(),
        'z_median': window_data['z_axis'].median(),
    }

    label = window_data["Label"].mode()[0]
    features["Label"] = label

    windows.append(features)

windowed_data = pd.DataFrame(windows)   # Turn list into pd.DataFrame object
saving_path = os.path.join(current_dir, "data", "processed", "dataset_val.csv")
windowed_data.to_csv(saving_path) 

# Merge the two dataframes to one on "TimeStamp"
dataframe_all = pd.merge_asof(dataframe_ciss, dataframe_registry, on="TimeStamp", direction="nearest")


# Define the maximum position of z axis while milling the pockets
peak_threshold = dataframe_all["Pos_Z"].max()
dataframe_all = dataframe_all[dataframe_all["Pos_Z"] >= peak_threshold]
dataframe_all.drop(columns={"Pos_X", "Pos_Y", "Pos_Z"}, inplace=True)


# Windowize the data
num_windows = 5400     # 36 pockets per tool, 3!!! tools, 50 windows for each pocket
window_size = len(dataframe_all) // num_windows
step_size = window_size

windows = []
for i in range(0, len(dataframe_all), step_size):
    if i + window_size > len(dataframe_all):
        break

    window_data = dataframe_all.iloc[i:i+window_size]

    # Feature extraction
    features = {
        'x_mean': window_data['x_axis'].mean(),
        'x_std': window_data['x_axis'].std(),
        'x_sum': window_data['x_axis'].sum(),
        'x_median': window_data['x_axis'].median(),
        'y_mean': window_data['y_axis'].mean(),
        'y_std': window_data['y_axis'].std(),
        'y_sum': window_data['y_axis'].sum(),
        'y_median': window_data['y_axis'].median(),
        'z_mean': window_data['z_axis'].mean(),
        'z_std': window_data['z_axis'].std(),
        'z_sum': window_data['z_axis'].sum(),
        'z_median': window_data['z_axis'].median(),
    }

    label = window_data["Label"].mode()[0]
    features["Label"] = label

    windows.append(features)

windowed_data = pd.DataFrame(windows)   # Turn list into pd.DataFrame object
saving_path = os.path.join(current_dir, "data", "processed", "dataset.csv")
windowed_data.to_csv(saving_path) 
