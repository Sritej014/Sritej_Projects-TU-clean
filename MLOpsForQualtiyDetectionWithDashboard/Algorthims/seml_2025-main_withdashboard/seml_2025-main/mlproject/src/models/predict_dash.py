import os
import pandas as pd
import time
import signal
import sys
import joblib
import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from pathlib import Path    
from dash import Dash, html, dash_table, dcc, callback, Output, Input
from matplotlib.pyplot import show
from dash.exceptions import PreventUpdate


# Input used while live testing
current_dir = Path(__file__).resolve().parents[2]
model_path = os.path.join(current_dir, "models", "svm_model.pkl")
scaler_path = os.path.join(current_dir, "models", "scaler.pkl")
scaler = joblib.load(scaler_path)
svm_model = joblib.load(model_path)


# Input used while live testing
# Streamdatapath is unique and linked to Florian Mitschke's local PC
#stream_data_path = r"C:\Users\F.Mitschke_Lokal\DIONE-X-PTW-Demonstrator\milling_machine_data\06012025_17_23_51"
#stream_data_path_reg = os.path.join(current_dir, "data", "data_reading_files")
#raw_data_folder = os.path.join(stream_data_path, "CISSRawData_LogFiles")  
#registry_folder = os.path.join(stream_data_path_reg, "WinPC_LogFiles")

# Input used for code testing
stream_data_path = os.path.join(current_dir, "data", "data_reading_files")
raw_data_folder = os.path.join(current_dir, "data", "raw", "Worn_tool_no2_pattern2", "CISSRawData_LogFilesTEST")
registry_folder = os.path.join(current_dir, "data", "raw", "Worn_tool_no2_pattern2" )
directory = raw_data_folder


# Count made predictions
count_predictions = 0
count_new = 0       
count_worn = 0
total_count = 0
good_tool_count = 0

# global variants
prediction_text = "No prediction yet..."
prediction_list = []
df_predictions_new = pd.DataFrame()
df_predictions = pd.DataFrame()
is_exiting = False

def signal_handler(sif, frame):
    # Handle KeyboardInterrupt to stop the loop.
    if count_new > count_worn:
        print("Exiting the loop...")
        print(f"New Tool Predictions made: {good_tool_count}")
        print(f"Predictions made: {total_count}")
        print("Final Prediction: New Tool")
        os._exit(0)

    elif count_new < count_worn:
        print("Exiting the loop...")
        print(f"Predictions made: {total_count}")
        print(f"Worn Tool Predictions made: {total_count - good_tool_count}")
        print("Final Prediction: Worn Tool")
        os._exit(0)
    else:
        print("No Tool Prediction was made...")
        os._exit(0)

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
       
external_stylesheets = [dbc.themes.UNITED]
app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = dbc.Container([
    dbc.Row([
        html.Div('ML Tutorium Live Data Analysis', className="text-primary text-left fs-2"),
    ]),
    dbc.Row([
        html.Div('Tool Status Prediction with SVM', className="text-dark text-left fs-3"),
    ]),
    dbc.Row([
        html.Div(id='prediction-text', style={'font-size': '24px'}),
    ]),
    dcc.Interval(
        id='interval-component',
        interval=25,  # in milliseconds (required Interval for 50 new entries)
        n_intervals=0
        )
], fluid=True)


@callback(
    [
    #Output('data_table', 'data'), 
    Output('prediction-text', 'children'),
    #Output('tool_status_pie_chart', 'figure'),
    ],
    [
    Input('interval-component', 'n_intervals')
    ]
)



def update_prediction(n):
    """Continuously monitor and process streaming data."""
    global count_predictions, count_new, count_worn, prediction_text, prediction_list, df_predictions_new, df_predictions, total_count, good_tool_count

    #if stop_loop:
    #    raise PreventUpdate
    

    
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
                    prediction_text = "Worn Tool"
                    print("Prediction: Worn Tool")
                elif prediction_counts.get(1, 0) < prediction_counts.get(0, 0):
                    count_new += 1
                    prediction_text = "New Tool"
                    print("Prediction: New Tool")
                else:                    
                        print("Uncertain Prediction")

                prediction_list.append({'Count': count_predictions, 'Prediction': prediction_text})

            except:
                print("Error fitting the data into the model.")
    
        #time.sleep(0.025)

        if prediction_list:
            df_predictions_new = pd.DataFrame(prediction_list)
            df_predictions = pd.concat([df_predictions, df_predictions_new], ignore_index=True)
            df_predictions = df_predictions.drop_duplicates(subset='Count', keep='first')
            df_predictions = df_predictions.sort_values(by='Count', ascending=True)
            good_tool_count = df_predictions[df_predictions['Prediction'] == 'New Tool'].shape[0]
            print(good_tool_count)
            total_count = df_predictions.shape[0]
            print(total_count)
            if good_tool_count > total_count /2:
                return [html.Span('New Tool', style={'color': 'green'})]
            else:
                return [html.Span('Worn Tool', style={'color': 'red'})]
     
    return [html.Span("No prediction yet...")]

# Run the app
if __name__ == '__main__':
    # Cleanup of old positions.txt and raw.txt files before starting the dashboard
    if os.path.exists("positions.txt"):
        os.remove("positions.txt")           
    if os.path.exists("raw.txt"):
        os.remove("raw.txt") 
    # Signal Handler before starting the dashboard to allow for controlled shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    app.run_server(debug=True)


