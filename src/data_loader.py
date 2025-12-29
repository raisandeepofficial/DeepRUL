import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import yaml

# --- SMART PATH RESOLUTION ---
# get the absolute path of the directory where this script lives
script_dir = os.path.dirname(os.path.abspath(__file__))
# go up one level to get the project root (DeepRUL/)
project_root = os.path.dirname(script_dir)
# construct absolute path to config
config_path = os.path.join(project_root, 'config', 'config.yaml')

# load config
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
else:
    raise FileNotFoundError(f"Config file not found at: {config_path}")
# -----------------------------

def load_data(mode='train', base_path=None):
    # columns based on cmapss documentation
    index_names = ['unit', 'time']
    setting_names = ['s_1', 's_2', 's_3']
    sensor_names = [f's{i}' for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names
    
    # determine file path
    file_name = cfg['train_file'] if mode == 'train' else cfg['test_file']
    
    # if base_path is not provided by the caller, use the config default relative to project root
    if base_path is None:
        base_path = os.path.join(project_root, 'data', 'raw')
        
    file_path = os.path.join(base_path, file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    print(f"Loading data from: {file_path}")

    # read csv
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=col_names)
    
    # rul calculation for training
    if mode == 'train':
        max_time = df.groupby('unit')['time'].max().reset_index()
        max_time.columns = ['unit', 'max']
        df = df.merge(max_time, on='unit', how='left')
        df['rul'] = df['max'] - df['time']
        df.drop('max', axis=1, inplace=True)
    
    return df

def get_data_loaders(data_path=None, batch_size=None, sequence_length=None):
    # allow overrides for flexibility (useful in notebooks)
    bs = batch_size if batch_size is not None else cfg['batch_size']
    seq_len = sequence_length if sequence_length is not None else cfg['sequence_length']
    
    train_df = load_data('train', base_path=data_path)
    
    # simple feature selection (dropping constant sensors for FD001)
    drop_cols = ['unit', 'time', 's_1', 's_2', 's_3', 's1', 's5', 's6', 's10', 's16', 's18', 's19']
    
    # scale data
    features = [c for c in train_df.columns if c not in drop_cols and c != 'rul']
    scaler = MinMaxScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    
    # sequence generation
    def gen_sequence(id_df, seq_length, seq_cols):
        data_matrix = id_df[seq_cols].values
        num_elements = data_matrix.shape[0]
        for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
            yield data_matrix[start:stop, :]
            
    def gen_labels(id_df, seq_length, label):
        data_matrix = id_df[label].values
        num_elements = data_matrix.shape[0]
        return data_matrix[seq_length:num_elements, :]

    x_train, y_train = [], []
    for unit in train_df['unit'].unique():
        unit_df = train_df[train_df['unit'] == unit]
        for seq in gen_sequence(unit_df, seq_len, features):
            x_train.append(seq)
        for label in gen_labels(unit_df, seq_len, ['rul']):
            y_train.append(label)
            
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    # verify dimensions match config
    if x_train.shape[2] != cfg['input_dim']:
        print(f"Warning: Config input_dim is {cfg['input_dim']} but data has {x_train.shape[2]}")
    
    train_data = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(train_data, shuffle=True, batch_size=bs)
    
    # dummy test loader (using same data for demo purposes, usually you'd load test_FD001)
    test_loader = DataLoader(train_data, shuffle=False, batch_size=bs)
    
    return train_loader, test_loader
