import os
import pandas as pd
import time
import signal
import sys
import joblib
from pathlib import Path    


current_dir = Path(__file__).resolve().parents[2]
model_path = os.path.join(current_dir, "models", "svm_model.pkl")
scaler_path = os.path.join(current_dir, "models", "scaler.pkl")
scaler = joblib.load(scaler_path)
svm_model = joblib.load(model_path)

stream_data_path = r"C:\Users\F.Mitschke_Lokal\DIONE-X-PTW-Demonstrator\milling_machine_data\06012025_17_23_51"
stream_data_path_reg = os.path.join(current_dir, "data", "data_reading_files")

raw_data_folder = os.path.join(stream_data_path, "CISSRawData_LogFiles")  
registry_folder = os.path.join(stream_data_path_reg, "WinPC_LogFiles")

# Count made predictions
count_predictions = 0
count_new = 0       
count_worn = 0

# Flag for stop/running condition
stop_loop = False

def signal_handler(sif, frame):
    """Handle KeyboardInterrupt to stop the loop."""
    global stop_loop
    print("Exiting the loop...")
    print(f"Predictions made: {count_predictions}")
    if count_new > count_worn:
        print("Final Prediction: New Tool")
    elif count_new < count_worn:
        print("Final Prediction: Worn Tool")
    stop_loop = True

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)


def get_last_csv(folder_path):
    csv_list = [csv for csv in os.listdir(folder_path) if csv.endswith(".csv")]
    csv_list_paths = [os.path.join(folder_path, csv) for csv in csv_list]
    csv_list_paths.sort(key=os.path.getmtime, reverse=True)
    last_csv_file = csv_list_paths[0]
    return last_csv_file


def read_new_rows_raw(folder_path, tracked_csv="raw.txt"):
    last_csv_file = get_last_csv(folder_path)
    if last_csv_file is None:
        print("No .csv files found.")
        return None
    
    last_csv_path = os.path.join(folder_path, last_csv_file)
    last_read_file = None
    last_read_position = 0

    if os.path.exists(tracked_csv):
        with open(tracked_csv, "r") as f:
            content = f.read().strip().split("\n")
            if len(content) == 2:
                last_read_file, last_read_position = content
                last_read_position = int(last_read_position)
    
    if last_read_file != last_csv_file:
        last_read_position = 0 # Starting from the first row 

    df_csv = pd.read_csv(last_csv_path,  sep = ",", skiprows=9, encoding='Windows-1252')

    if last_read_position < len(df_csv):
        new_rows = df_csv.iloc[last_read_position:]
        new_rows = pd.concat([df_csv.iloc[:1], new_rows], ignore_index=True)    # Include header
        with open(tracked_csv, "w") as f:
            f.write(f"{last_csv_file}\n{len(df_csv)}")
        return new_rows
    else:
        print("No new rows found (raw file).")
        return None


def read_new_rows(folder_path, tracked_csv="positions.txt"):
    last_csv_file = get_last_csv(folder_path)
    if last_csv_file is None:
        print("No .csv files found.")
        return None
    
    last_csv_path = os.path.join(folder_path, last_csv_file)
    last_read_file = None
    last_read_position = 0

    if os.path.exists(tracked_csv):
        with open(tracked_csv, "r") as f:
            content = f.read().strip().split("\n")
            if len(content) == 2:
                last_read_file, last_read_position = content
                last_read_position = int(last_read_position)
    
    if last_read_file != last_csv_file:
        last_read_position = 0 # Starting from the first row 

    df_csv = pd.read_csv(last_csv_path)

    if last_read_position < len(df_csv):
        new_rows = df_csv.iloc[last_read_position:]
        new_rows = pd.concat([df_csv.iloc[:1], new_rows], ignore_index=True)    # Include header
        with open(tracked_csv, "w") as f:
            f.write(f"{last_csv_file}\n{len(df_csv)}")
        return new_rows
    else:
        print("No new rows found (registry file).")
        return None
    

def extract_features(data, window_size=50):
    if not data is None:
        data.rename(columns={"Unix TimeStamp (ms)": "TimeStamp", "Data / x_axis": "x_axis"}, inplace=True)
        data = data.dropna()
    else: return None

    windows = []
    for i in range(0, len(data) - window_size + 1, window_size):  # Step by window_size
        window_data = data.iloc[i:i + window_size]

        # Extract features for the window
        features = {
            "x_mean": window_data["x_axis"].mean(),
            "x_std": window_data["x_axis"].std(),
            "x_sum": window_data["x_axis"].sum(),
            "x_median": window_data["x_axis"].median(),
            "y_mean": window_data["y_axis"].mean(),
            "y_std": window_data["y_axis"].std(),
            "y_sum": window_data["y_axis"].sum(),
            "y_median": window_data["y_axis"].median(),
            "z_mean": window_data["z_axis"].mean(),
            "z_std": window_data["z_axis"].std(),
            "z_sum": window_data["z_axis"].sum(),
            "z_median": window_data["z_axis"].median(),
        }
       
        windows.append(features)

    # Convert list of features into a DataFrame
    windowed_df = pd.DataFrame(windows)
    return windowed_df

        
def main_loop():
    """Continuously monitor and process streaming data."""
    global stop_loop, count_predictions, count_new, count_worn

    while not stop_loop:
        last_registry_rows = read_new_rows(registry_folder)
        if not last_registry_rows is None:
            rows_pos_z = last_registry_rows[last_registry_rows["Pos_Z"] >= 42062]
            if not rows_pos_z.empty:
                last_raw_rows = read_new_rows_raw(raw_data_folder)
                features_df = extract_features(last_raw_rows)
                #rows_of_data = pd.DataFrame()
                try:
                    features_df_scaled = scaler.transform(features_df)
                    prediction = svm_model.predict(features_df_scaled)
                    count_predictions += 1
                    prediction_counts = pd.DataFrame(prediction).iloc[:, 0].value_counts()
                    if prediction_counts.get(1, 0) > prediction_counts.get(0, 0):
                        count_worn += 1
                        print("Prediction: Worn Tool")
                    elif prediction_counts.get(1, 0) < prediction_counts.get(0, 0):
                        count_new += 1
                        print("Prediction: New Tool")
                    else:                    
                        print("Uncertain Prediction")
                except:
                    print("Error fitting the data into the model.")
            
        time.sleep(0.025)


if __name__ == "__main__":
    print("Starting the streaming prediction process. Press Ctrl+C to stop.")
    '''try:
        main_loop()
    except Exception as e:
        print(f"Error reading the data stream: {e}")'''
    main_loop()
    if os.path.exists("positions.txt"):
        os.remove("positions.txt")
    if os.path.exists("raw.txt"):
        os.remove("raw.txt")
