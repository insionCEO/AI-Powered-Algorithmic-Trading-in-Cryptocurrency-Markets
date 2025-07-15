import json

import pandas as pd


# def extract_time_features_old(df, lowest_time_feature='M'):
#     # Extract time features based on the lowest_time_feature parameter
#     # Minute Data
#     if lowest_time_feature == 'Min':
#         time_features = ['month', 'day', 'weekday', 'hour', 'minute']
#     # Seconds data
#     elif lowest_time_feature == 'Sec':
#         time_features = ['month', 'day', 'weekday', 'hour', 'minute', 'second']
#     else:
#         raise ValueError("lowest_time_feature parameter must be 'M' or 'S'")
#
#     # Extract time features using the pandas dt accessor
#     df_time = pd.DataFrame()
#     for feature in time_features:
#         df_time[feature] = df['date'].dt.__getattribute__(feature)
#
#     # Convert time features to a NumPy array
#     date_features = df_time.values
#
#     return date_features

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


def save_args_to_text(args, file_path):
    """
    Save experiment arguments to a text file.
    """
    with open(file_path, 'w') as f:
        for key, value in args.__dict__.items():
            if not key.startswith('__') and not key.endswith('__'):
                f.write(f"{key}: {value}\n")


def load_args_from_text(file_path):
    """
    Load experiment arguments from a text file.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    args = DotDict()

    for line in lines:
        key, value = line.strip().split(': ')
        args[key] = value

    return args
