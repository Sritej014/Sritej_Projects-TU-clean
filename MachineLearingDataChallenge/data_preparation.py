import pandas as pd
import numpy as np
import h5py
import json
from pathlib import Path


def read_h5(path):
    """
    Function to read HDF5 files.
    Return data and column names.
    """
    with h5py.File(path, 'r') as file:
        data = file['data'][:]
        column_names = file['data'].attrs['column_names']
        df = pd.DataFrame(data, columns=column_names)
        df['timestamp'] = df['timestamp'] / 1000000 # make sure unit='s'
    return df


def read_data(part_id, json_path, sensor_path, side, process=None):
    check_dict = {
        'frontside': [
            'face_milling',
            'outer_contour_roughing_and_finishing',
            'lateral_groove',
            'stepped_bore',
            'outer_contour_deburring_holes',
            'lateral_drilling',
            'drilling_countersinking',
            'drilling',
            'thread_miling'
        ],
        'backside': [
            'face_milling',
            'circular_pocket_milling',
            'component_deburring',
            'ring_groove'
        ]
    }

    part_id = str(part_id)

    # check if 'side' in keys of check_dict
    if side not in check_dict:
        raise KeyError(f"Wrong side: {side}")
    
    # check if 'process' in values
    if process:
        if process not in check_dict[side]:
            raise ValueError(f"Wrong process: {process}")
    
    # load json file
    json_file_path = Path(json_path)

    # check if the path exist
    if not json_file_path.exists():
        raise FileNotFoundError(f"Json file does not exist: {json_file_path}")

    # load json file
    with open(json_file_path, 'r') as jf:
        dicts = json.load(jf)

    
    data_dict = {}
    for dict in dicts:
        if dict['part_id'] == part_id:
            data_dict['part_id'] = dict['part_id']
            data_dict['start_time'] = dict['process_data'][1]['start_time']
            data_dict['end_time'] = dict['process_data'][1]['end_time']
            data_dict['anomaly'] = dict['process_data'][1]['anomaly']


            # handle 'SENSOR_FAILURE'
            files = dict['process_data'][1]['data_paths']
            for file in files:
                if 'SENSOR_FAILURE' in file:
                    print(f"Sensor failure for part_id={data_dict['part_id']}")
                    return
                    
            # filter data path
            files = dict['process_data'][1]['data_paths']
            path_file = [file for file in files if side in file and 'external' in file and file.endswith('.h5')]
            if len(path_file) != 1:
                raise ValueError(f"Error by filtering file: {path_file}")
            
            split_string = path_file[0].split('/')
            data_dict['data_path'] = sensor_path + split_string[-2] + '/' + split_string[-1] 


    df = read_h5(data_dict['data_path'])
    df.dropna()

    # check the correctness of timestamp
    if (df['timestamp'].iloc[0] < data_dict['start_time']) or (df['timestamp'].iloc[-1] > data_dict['end_time']):
        raise ValueError(f"Timestamp wrong.")

    df['total_acc'] = np.sqrt((df[['acc_x', 'acc_y', 'acc_z']]**2).sum(axis=1))
    
    df['part_id'] = data_dict['part_id']


    return df

def custom_metric(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return (precision, recall, f1)