
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


from drl_strategy import *
from portfolio import Portfolio
from data import HistoricCSVDataHandler
from transformer_predictor import TransformerPredictor


# Create an instance of the Transformer_Predict class
path_to_model = "models/"
# model = "informer_BTC_ftS_sl180_ll60_pl30_dm1024_nh2_el2_dl8_df1024_atprob_fc5_ebtimeF_dtTrue_mxTrue_exp_1"
# model = "transformer_BTC_ftS_sl180_ll60_pl30_dm512_nh16_el2_dl6_df1024_atprob_fc5_ebtimeF_dtTrue_mxTrue_exp_1"
model = "pyraformer_final_180_60_30"
path_to_checkpoint = os.path.join(path_to_model, model, "checkpoint.pth")
path_to_args_txt = os.path.join(path_to_model, model, "args.txt")
input_sequence_length = 180
label_length = 60  # how much past data to feed into the decoder
pred_length = 30  # output sequence length # ONLY CHANGE THIS
device = "cpu"

strategy = TransformerPredictor(seq_length=input_sequence_length,
                                label_len=label_length,
                                pred_len=pred_length,
                                # data_df=None,
                                path_to_args_txt=path_to_args_txt,
                                path_to_checkpoints=path_to_checkpoint,
                                device=device)
strategy.load_model_pred()

data_dir = os.path.join(os.getcwd(), '..', 'data/processed/10-min-samples/')
df_raw = pd.read_csv(os.path.join(data_dir, 'sample-1-BTC.csv'))
df_raw.head()
num_points = 240
extracted_df = df_raw[:num_points]
print(extracted_df)


file_path = "../data/processed/10-min-samples/sample-1-BTC.csv"
data_handler = HistoricCSVDataHandler(file_path)
portfolio = Portfolio(data_handler)

drlStrategy = DRLModel(0)
drlStrategy.load_model("models/drl/")


strategy_dict = {"portfolio_cash_balance": portfolio.current_holdings['cash'],
                 "portfolio_current_stock": portfolio.current_positions['BTC'],
                 "portfolio_max_stock": portfolio.max_stock,
                 "portfolio_max_percentile_investment": portfolio.max_investment_per_trade,
                 "portfolio_initial_investment": portfolio.initial_capital,
                 }


start_time = time.time()
signals = strategy.calculate_signals(extracted_df)


prediction_data = signals.values.tolist()
past_data = extracted_df.values.tolist()
# Get only the last 2 columns from past_data = midpoint and spread
past_data_ = [row[-2:] for row in past_data]
# Get only the last column from prediction_data = prices
prediction_data_ = [row[-1] for row in prediction_data]


action = drlStrategy.get_prediction(
    past_data_, prediction_data_, strategy_dict)

end_time = time.time()

execution_time = end_time - start_time

print(f"Execution time for calculate_signals: {execution_time:.4f} seconds")


print("SIGNALS: ")
print(signals)

print("ACTION: ")
print(action)
