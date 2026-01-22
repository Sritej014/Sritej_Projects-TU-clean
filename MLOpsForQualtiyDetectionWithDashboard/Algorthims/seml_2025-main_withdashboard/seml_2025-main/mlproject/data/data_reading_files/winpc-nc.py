import csv
import os, sys
import time
import winreg
from datetime import datetime

# Get the path of the script and create the folder if it doesn't exist
scriptPath = os.path.dirname(os.path.abspath(sys.argv[0]))
log_folder = scriptPath + "/WinPC_LogFiles"
if not os.path.exists(log_folder): 
    os.makedirs(log_folder)

# Define the registry key and values
HIVE = winreg.HKEY_CURRENT_USER
SUBKEY = r"SOFTWARE\BLSoft\WinPC-NC"
REGISTRY_VALUES = [
    "WinPCNCRunning",
    "CurrentJobProgress",
    "CurrentJobTimeMin",
    "CurrentJobTimeSec",
    "CurrentJobCommandNo",
    "CurrentSpdOvr",
    "State",
    "Pos_X",
    "Pos_Y",
    "Pos_Z"
]

# CSV File Configuration
LOG_FILE_PATH = f"{log_folder}/WINPC_NC_Log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
MAX_ENTRIES = 25000  # Max entries before creating a new file
entry_counter = 0  # Counter for entries


def read_registry_value(hive, subkey, value_name):
    """Reads a value from the Windows Registry."""
    try:
        # Open the registry key
        key = winreg.OpenKey(hive, subkey, 0, winreg.KEY_READ)

        # Read the registry value
        value, _ = winreg.QueryValueEx(key, value_name)

        # Close the registry key
        winreg.CloseKey(key)

        return float(value)
    
    except FileNotFoundError:
        print("Registry key not found.")
        return "REGISTRY NOT FOUND"
    except OSError:
        print("Error accessing the registry.")
        return "ACCESS ERROR"



def write_to_csv(log_file_path, data):
    """Writes data to a CSV file."""
    global entry_counter

    # Initialize file if it doesn't exist
    if entry_counter == 0:
        with open(log_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, dialect='excel')
            csv_writer.writerow(["sep=,"])  # CSV Separator
            csv_writer.writerow(["WINPC-NC Process Log"])
            csv_writer.writerow(["Generated on", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            csv_writer.writerow(["This log captures milling process data from WINPC-NC via Windows Registry."])
            csv_writer.writerow([" "])
            csv_writer.writerow(["Timestamp"] + REGISTRY_VALUES)
    
    # Append data to the CSV
    with open(log_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, dialect='excel')
        csv_writer.writerow(data)

    entry_counter += 1

    # Reset counter and file path if max entries are reached
    if entry_counter >= MAX_ENTRIES:
        print("Max entries reached. Starting a new file.")
        global LOG_FILE_PATH
        LOG_FILE_PATH = f"{log_folder}/WINPC_NC_Log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        entry_counter = 0


def main():
    """Main function to read registry values and log them."""
    while True:
        try:
            # Get the current timestamp
            #timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            timestamp = int(time.time()*1000)

            # Read registry values
            data = [timestamp]
            for value_name in REGISTRY_VALUES:
                value = read_registry_value(HIVE, SUBKEY, value_name)
                data.append(value)

            # Write data to CSV
            write_to_csv(LOG_FILE_PATH, data)

            # Delay before the next read
            time.sleep(0.0005)  # Adjust the interval as needed

        except KeyboardInterrupt:
            print("Logging stopped by user.")
            break


if __name__ == "__main__":
    main()
