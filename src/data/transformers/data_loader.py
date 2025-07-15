import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from .utils import extract_time_features


class Dataset_Train_Val(Dataset):
    def __init__(self, root_path, data_path, flag='train', size=None,
                 features='S', target='midpoint', normalise=True, freq='Min', timestamp=True):

        # size array -> [seq_len, label_len, pred_len]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        # Using assert to make sure only train and val keywords are used
        assert flag in ['train', 'val']
        type_map = {'train': 0, 'val': 1}
        self.set_type = type_map[flag]

        self.features = features  # S or M refers to the fact if we are using Multi variate data or Single Variate data
        self.target = target  # Prediction variable I.e Price
        self.normalise = normalise  # If we are normalising the train data or not

        self.time_freq = freq  # time frequency of the data -> [Min or Sec]
        self.timestamp = timestamp  # set to true if the date column is a timestamp

        # Path and file name of the data
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        # If data has time stamps convert it to a date and remove the timestamp col
        if self.timestamp:
            df_raw['date'] = pd.to_datetime(df_raw['timestamp'], unit='ms')
            df_raw = df_raw.drop(columns=['timestamp'])

        '''
         df_raw.columns: ['date', ...(other features), target feature]
        '''
        # Format the data
        columns = list(df_raw.columns)
        columns.remove(self.target)
        columns.remove('date')
        df_raw = df_raw[['date'] + columns + [self.target]]

        # Break it down into train and validation sets
        train_size = int(len(df_raw) * 0.8)
        val_size = len(df_raw) - train_size

        # Borders are used to split the data border 1 is the start and border 2 is the end
        border1s = [0, train_size - self.seq_len]
        border2s = [train_size, len(df_raw)]

        # Decide on the start and the end depending on the type which is 0 for train or 1 val sets
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Multivariate or Single Variate data split on S only date and target variable is used
        if self.features == 'MS':
            # TO DO: set which columns we want to use for Multi-Variate data
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # Normalise the training data if it's enabled
        if self.normalise:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # select the 'date' column of the input DataFrame
        df_stamp = df_raw[['date']][border1:border2]

        # convert the 'date' column to datetime format
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        date_features = extract_time_features(df_stamp, self.time_freq)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.date_features = date_features

    def __getitem__(self, index):
        # Calculate start and end indices for the input sequence
        s1_start_idx = index
        s1_end_idx = s1_start_idx + self.seq_len

        # Calculate start and end indices for the output sequence (labels + predictions)
        s2_start_idx = s1_end_idx - self.label_len
        s2_end_idx = s2_start_idx + self.label_len + self.pred_len

        seq_x = self.data_x[s1_start_idx:s1_end_idx]
        seq_y = self.data_y[s2_start_idx:s2_end_idx]

        # Extract input and output time features (timestamp) from the data
        seq_x_date = self.date_features[s1_start_idx:s1_end_idx]
        seq_y_date = self.date_features[s2_start_idx:s2_end_idx]

        return seq_x, seq_y, seq_x_date, seq_y_date

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def get_scaler(self):
        return self.scaler


class Dataset_Test(Dataset):
    def __init__(self, root_path, data_path, size=None, flag='test',
                 features='S', target='midpoint', freq='Min', normalise=False, timestamp=True):

        # size array -> [seq_len, label_len, pred_len]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.features = features  # S or M refers to the fact if we are using Multi variate data or Single Variate data
        self.target = target  # Prediction variable I.e Price
        self.normalise = normalise  # If we are normalising the test data or not

        # time frequency of the data -> [Min or Sec]
        self.time_freq = freq
        self.timestamp = timestamp

        # Path and file name of the data
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        # If data has time stamps convert it to a date and remove the timestamp col
        if self.timestamp:
            df_raw['date'] = pd.to_datetime(df_raw['timestamp'], unit='ms')
            df_raw = df_raw.drop(columns=['timestamp'])

        '''
         df_raw.columns: ['date', ...(other features), target feature]
        '''
        # Format the data
        columns = list(df_raw.columns)
        columns.remove(self.target)
        columns.remove('date')
        df_raw = df_raw[['date'] + columns + [self.target]]

        # Multivariate or Single Variate data split on S only date and target variable is used
        if self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.normalise:
            test_data = df_data
            self.scaler.fit(test_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # select the 'date' column of the input DataFrame
        df_stamp = df_raw[['date']]

        # convert the 'date' column to datetime format
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        date_features = extract_time_features(df_stamp, self.time_freq)

        self.data_x = data
        self.data_y = data
        self.date_features = date_features

    def __getitem__(self, index):
        # Calculate start and end indices for the input sequence
        s1_start_idx = index
        s1_end_idx = s1_start_idx + self.seq_len

        # Calculate start and end indices for the output sequence (labels + predictions)
        s2_start_idx = s1_end_idx - self.label_len
        s2_end_idx = s2_start_idx + self.label_len + self.pred_len

        seq_x = self.data_x[s1_start_idx:s1_end_idx]
        seq_y = self.data_y[s2_start_idx:s2_end_idx]

        # Extract input and output time features (timestamp) from the data
        seq_x_date = self.date_features[s1_start_idx:s1_end_idx]
        seq_y_date = self.date_features[s2_start_idx:s2_end_idx]

        return seq_x, seq_y, seq_x_date, seq_y_date

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def get_scaler(self):
        return self.scaler


class LiveDataLoader(Dataset):
    def __init__(self, data_df, size=None, flag='test',
                 features='S', target='midpoint', freq='Min', normalise=False, timestamp=True):

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.features = features  # S or M refers to the fact if we are using Multi variate data or Single Variate data
        self.target = target  # Prediction variable I.e Price
        self.normalise = normalise  # If we are normalising the test data or not

        # time frequency of the data -> [Min or Sec]
        self.time_freq = freq
        self.timestamp = timestamp

        self.__read_data__(data_df)

    def __read_data__(self, data_df):
        self.scaler = StandardScaler()
        df_raw = data_df

        if self.timestamp:
            df_raw['date'] = pd.to_datetime(df_raw['timestamp'], unit='ms')
            df_raw = df_raw.drop(columns=['timestamp'])

        '''
         df_raw.columns: ['date', ...(other features), target feature]
        '''
        # Format the data
        columns = list(df_raw.columns)
        columns.remove(self.target)
        columns.remove('date')
        df_raw = df_raw[['date'] + columns + [self.target]]

        # Multivariate or Single Variate data split on S only date and target variable is used
        if self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.normalise:
            test_data = df_data
            self.scaler.fit(test_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # select the 'date' column of the input DataFrame
        df_stamp = df_raw[['date']]

        # convert the 'date' column to datetime format
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        date_features = extract_time_features(df_stamp, self.time_freq)

        self.data_x = data
        self.data_y = data
        self.date_features = date_features

    def __getitem__(self, index):
        # Calculate start and end indices for the input sequence
        s1_start_idx = index
        s1_end_idx = s1_start_idx + self.seq_len

        # Calculate start and end indices for the output sequence (labels + predictions)
        s2_start_idx = s1_end_idx - self.label_len
        s2_end_idx = s2_start_idx + self.label_len + self.pred_len

        seq_x = self.data_x[s1_start_idx:s1_end_idx]
        seq_y = self.data_y[s2_start_idx:s2_end_idx]

        # Extract input and output time features (timestamp) from the data
        seq_x_date = self.date_features[s1_start_idx:s1_end_idx]
        seq_y_date = self.date_features[s2_start_idx:s2_end_idx]

        return seq_x, seq_y, seq_x_date, seq_y_date

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def get_scaler(self):
        return self.scaler
