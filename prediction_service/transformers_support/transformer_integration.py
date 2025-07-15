import numpy as np
import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from transformers.models import Informer ,Transformer ,Pyraformer ,ETSformer


def extract_time_features(df, lowest_time_feature='Sec'):
    # Define the maximum value for each time feature
    max_month = 12.0
    max_day = 31.0
    max_weekday = 6.0
    max_hour = 23.0
    max_minute = 59.0
    max_second = 59.0

    # Extract time features based on the lowest_time_feature parameter
    if lowest_time_feature == 'Min':
        time_features = ['month', 'day', 'weekday', 'hour', 'minute']
        max_features = [max_month, max_day, max_weekday, max_hour, max_minute]
    elif lowest_time_feature == 'Sec':
        time_features = ['month', 'day', 'weekday', 'hour', 'minute', 'second']
        max_features = [max_month, max_day, max_weekday, max_hour, max_minute, max_second]
    else:
        raise ValueError("lowest_time_feature parameter must be 'Min' or 'Sec'")

    # Extract time features using the pandas dt accessor
    df_time = pd.DataFrame()
    for feature, max_feature in zip(time_features, max_features):
        df_time[feature] = df['date'].dt.__getattribute__(feature) / max_feature - 0.5

    # Convert time features to a NumPy array
    date_features = df_time.values

    return date_features


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

        # select the 'date' column of the input DataFrame and create a copy
        df_stamp = df_raw[['date']].copy()

        # convert the 'date' column to datetime format using .loc[]
        df_stamp.loc[:, 'date'] = pd.to_datetime(df_stamp['date'])

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


def select_criterion():
    criterion = nn.MSELoss()
    return criterion


def build_model(args):
    model_dict = {
        'informer': Informer,
        'transformer': Transformer,
        'pyraformer' : Pyraformer,
        'etsformer' : ETSformer
    }

    if args.model == 'informer' or args.model == 'transformer':
        model = model_dict[args.model](
            args.enc_in,
            args.dec_in,
            args.c_out,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.factor,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.dropout,
            args.attn,
            args.embed,
            args.freq,
            args.activation,
            args.output_attention,
            args.distil,
            args.mix,
            args.device
        ).float()
    elif args.model == 'pyraformer':
        model = model_dict[args.model](
            args.enc_in,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_ff,
            args.dropout,
            args.window_size,
            args.inner_size
        ).float()
    elif args.model == 'etsformer':
        model = model_dict[args.model](
            args.enc_in,
            args.c_out,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.dropout,
            args.embed,
            args.freq,
            args.activation,
            args.top_k
        ).float()

    return model


class DotDict(dict):
    """
    Dot notation access to dictionary attributes used for the args dictionary
    """

    def __getattr__(self, attr):
        return self.get(attr)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self.update(d)


def load_args_from_text(file_path):
    """
    Load experiment arguments from a text file.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    args = DotDict()

    for line in lines:
        key, value = line.strip().split(': ')
        if key == 'window_size':
            value = [int(x.strip()) for x in value.strip('[]').split(',')]
        else:
            # Try to convert value to int, float, or bool if possible
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value == 'True':
                        value = True
                    elif value == 'False':
                        value = False
        args[key] = value

    return args
