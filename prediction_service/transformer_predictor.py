from __future__ import print_function
import time
from abc import ABCMeta, abstractmethod
import datetime

import numpy as np
import pandas as pd
import os

# transformers
import torch
from torch.utils.data import DataLoader
from transformers_support.transformer_integration import load_args_from_text, \
    LiveDataLoader, build_model

from portfolio import Portfolio
from data import HistoricCSVDataHandler


class TransformerPredictor(object):

    def __init__(self, seq_length, label_len, pred_len, path_to_args_txt=None,
                 path_to_checkpoints=None, device='cpu'):

        # self.num_signals = 0

        self.seq_length = seq_length
        self.label_len = label_len
        self.pred_len = pred_len
        self.path_to_checkpoints = path_to_checkpoints
        self.device = device
        self.model_loaded = False  # added flag to track whether the model has been loaded
        self.model = None

        # Load arguments from text file
        if path_to_args_txt is not None:
            self.args = load_args_from_text(path_to_args_txt)
            self.args.seq_length = self.seq_length
            self.args.label_len = self.label_len
            self.args.pred_len = self.pred_len
            self.args.device = self.device

    def _generate_next_points(self, data_df, extra_points):
        # Sort the input dataframe by timestamp in ascending order
        # data_df = data_df.sort_values(by='timestamp')

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
        next_midpoints = np.array(
            [last_midpoints[-1] + delta_midpoint * (i + 1) for i in range(extra_points)])
        next_spreads = np.array(
            [last_spreads[-1] + delta_spread * (i + 1) for i in range(extra_points)])

        # Compute the timestamps of the next set of points to be generated
        last_timestamp = last_n_observations['timestamp'].iloc[-1]
        next_timestamps = pd.date_range(start=pd.Timestamp(last_timestamp, unit='ms') + pd.Timedelta(seconds=1),
                                        periods=extra_points, freq='1S')

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
        extra_points = self.pred_len

        input_df = self._generate_next_points(data_df, extra_points)

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
            model.load_state_dict(torch.load(
                self.path_to_checkpoints, map_location=torch.device(self.device)))

            model.to(self.device)

            self.model = model  # assign model to self for future use
            self.model_loaded = True  # set flag to indicate that the model has been loaded

        return self.model

    def calculate_signals(self, data_df):
        # Check if dataframe length matches sequence length
        if data_df is not None:
            if len(data_df) != self.seq_length + self.label_len:
                raise ValueError(
                    f"Dataframe length {len(data_df)} does not match sequence length {self.seq_length}")

        # Load the prediction data
        data_set, data_loader, input_df = self._read_pred_data(data_df)

        # Load the trained model
        model = self.model

        preds = []

        model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
                print(i)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(
                    batch_y[:, -self.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = model(batch_x, batch_x_mark,
                                            dec_inp, batch_y_mark)[0]
                        else:
                            outputs = model(batch_x, batch_x_mark,
                                            dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = model(batch_x, batch_x_mark,
                                        dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark,
                                        dec_inp, batch_y_mark)

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
        unscaled_predictions = scaler.inverse_transform(
            predictions.reshape(-1, 1))

        # Reshape the unscaled predictions to their original shape
        unscaled_predictions = unscaled_predictions.reshape(
            predictions.shape[:-1])
        df_prices = pd.DataFrame(unscaled_predictions, columns=['Price'])

        # Extract the last pred_len rows from the input_df DataFrame
        input_last = input_df.iloc[-self.args.pred_len:, :]["date"]

        # Combine the two DataFrames into a single DataFrame
        df_combined = pd.concat(
            [input_last.reset_index(drop=True), df_prices], axis=1)

        # Rename the columns to match the expected output format
        df_combined = df_combined.rename(
            columns={'date': 'Date', 'Price': 'price'})

        return df_combined
