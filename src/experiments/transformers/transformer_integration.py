import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from models.transformers.models import Informer


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


def build_model(args):
    model_dict = {
        'informer': Informer,
    }

    if args.model == 'informer':
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


class Transformer_Predict:
    def __init__(self, seq_length, label_len, pred_len, data_df=None, path_to_args_txt=None,
                 path_to_checkpoints=None, device='cpu'):
        self.seq_length = seq_length
        self.label_len = label_len
        self.pred_len = pred_len
        self.path_to_checkpoints = path_to_checkpoints
        self.device = device
        self.data_df = data_df
        self.model_loaded = False  # added flag to track whether the model has been loaded

        # Load arguments from text file
        if path_to_args_txt is not None:
            self.args = load_args_from_text(path_to_args_txt)
            self.args.seq_length = self.seq_length
            self.args.label_len = self.label_len
            self.args.pred_len = self.pred_len

        # Check if dataframe length matches sequence length
        if data_df is not None:
            if len(self.data_df) != self.seq_length:
                raise ValueError(
                    f"Dataframe length {len(self.data_df)} does not match sequence length {self.seq_length}")

    def _generate_next_points(self, extra_points):
        # Sort the input dataframe by timestamp in ascending order
        data_df = self.data_df.sort_values(by='timestamp')

        # Extract the last n observations from the input dataframe
        n = 10
        last_n_observations = data_df.tail(n)

        # Extract the midpoint and spread values from the last n observations
        last_midpoints = last_n_observations['midpoint'].values
        last_spreads = last_n_observations['spread'].values

        # Compute the change in midpoint and spread from the last observation to the first generated point
        delta_midpoint = np.random.normal(loc=0, scale=0.1)
        delta_spread = np.random.normal(loc=0, scale=0.05)

        # Generate the next set of points
        next_midpoints = np.array([last_midpoints[-1] + delta_midpoint * (i + 1) for i in range(extra_points)])
        next_spreads = np.array([last_spreads[-1] + delta_spread * (i + 1) for i in range(extra_points)])

        # Compute the timestamps of the next set of points to be generated
        last_timestamp = last_n_observations['timestamp'].iloc[-1]
        next_timestamps = pd.date_range(start=pd.Timestamp(last_timestamp, unit='ms') + pd.Timedelta(milliseconds=100),
                                        periods=extra_points, freq='100L')

        # convert the column of dates to Unix timestamps
        next_timestamps = next_timestamps.astype(np.int64) // 10 ** 6

        # Construct a new dataframe with the generated points
        next_df = pd.DataFrame({
            'timestamp': next_timestamps,
            'midpoint': next_midpoints,
            'spread': next_spreads
        })

        new_generated = pd.concat([data_df, next_df], ignore_index=True)

        return new_generated

    def _args_as_dotdict(self):
        return self.args

    def _read_pred_data(self, data_df):
        extra_points = self.pred_len + self.label_len

        input_df = self._generate_next_points(extra_points)

        data_set = LiveDataLoader(
            data_df=input_df,
            size=[self.seq_length, self.label_len, self.pred_len],
            features=self.args['features'],
            target=self.args['target'],
            normalise=self.args['normalise'],
            freq=self.args['freq'],
            timestamp=self.args['timestamp']
        )

        data_loader = DataLoader(
            data_set,
            batch_size=self.label_len,  # set it to 1 to not miss out on any input
            shuffle=False,
            num_workers=int(self.args['num_workers']),
            drop_last=True)

        return data_set, data_loader, input_df

    def load_model_pred(self):
        if not self.model_loaded:  # check if the model has already been loaded
            model = build_model(self.args)

            print('loading model')
            model.load_state_dict(torch.load(self.path_to_checkpoints))

            model.to(self.device)

            self.model = model  # assign model to self for future use
            self.model_loaded = True  # set flag to indicate that the model has been loaded

        return self.model

    def predictions(self, data_df):
        # Load the prediction data
        data_set, data_loader, input_df = self._read_pred_data(data_df)

        # Load the trained model
        model = self.model

        preds = []

        model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
                print(i)
                batch_x = batch_x.float().to(self.args.device)
                batch_y = batch_y.float().to(self.args.device)

                batch_x_mark = batch_x_mark.float().to(self.args.device)
                batch_y_mark = batch_y_mark.float().to(self.args.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.args.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.pred_len:, f_dim:]
                outputs = outputs.detach().cpu().numpy()

                pred = outputs

                preds.append(pred)

        predictions_array = np.array(preds)

        # Obtain the Scaler object from data_set
        scaler = data_set.get_scaler()

        # Obtain the predicted values
        predictions = predictions_array[0][0]

        # Invert the transformation applied to the predicted values
        unscaled_predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

        # Reshape the unscaled predictions to their original shape
        unscaled_predictions = unscaled_predictions.reshape(predictions.shape[:-1])
        df_prices = pd.DataFrame(unscaled_predictions, columns=['Price'])

        # Extract the last pred_len rows from the input_df DataFrame
        input_last = input_df.iloc[-self.args.pred_len:, :]["date"]

        # Combine the two DataFrames into a single DataFrame
        df_combined = pd.concat([input_last.reset_index(drop=True), df_prices], axis=1)

        # Rename the columns to match the expected output format
        df_combined = df_combined.rename(columns={'date': 'Date', 'Price': 'price'})

        return df_combined
