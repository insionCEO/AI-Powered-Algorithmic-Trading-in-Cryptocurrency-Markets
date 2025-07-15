import asyncio
import json
import pandas as pd
import numpy as np
import websockets
from typing import Dict, Any
import os
import multiprocessing
import pickle
import concurrent.futures
import time
import threading

from data import HistoricCSVDataHandler
from execution import SimulatedExecutionHandler
from portfolio import Portfolio
from transformer_predictor import TransformerPredictor
from drl_strategy import *


class MLHandler:

    def __init__(self):
        self.informer = self.load_transformer(
            "informer_BTC_ftS_sl180_ll60_pl30_dm1024_nh2_el2_dl8_df1024_atprob_fc5_ebtimeF_dtTrue_mxTrue_exp_1", 180, 60, 30)  # possible TODO read transformer parameters from args.txt (which shoulds have been .yaml to start with)
        self.pyraformer = self.load_transformer(
            "pyraformer_final_180_60_30", 180, 60, 30)
        self.transformer = self.load_transformer(
            "transformer_BTC_ftS_sl180_ll60_pl30_dm512_nh16_el2_dl6_df1024_atprob_fc5_ebtimeF_dtTrue_mxTrue_exp_1", 180, 60, 30)

        self.drl = self.load_drl()

    def load_transformer(self, model_name, input_sequence_length, label_length, pred_length):
        # Create an instance of the Transformer_Predict class
        path_to_model = "models/"
        path_to_checkpoint = os.path.join(
            path_to_model, model_name, "checkpoint.pth")
        path_to_args_txt = os.path.join(path_to_model, model_name, "args.txt")
        device = "cpu"

        transformer_predictor = TransformerPredictor(seq_length=input_sequence_length, label_len=label_length,
                                                     pred_len=pred_length,
                                                     path_to_args_txt=path_to_args_txt,
                                                     path_to_checkpoints=path_to_checkpoint,
                                                     device=device)

        transformer_predictor.load_model_pred()

        print("DEBUG: LOADED TRANSFORMER")
        return transformer_predictor

    def load_drl(self):

        drlStrategy = DRLModel(0)
        drlStrategy.load_model("models/drl/")

        return drlStrategy

    def predict(self, conn, df, strategy_dict):

        try:
            # start_time = time.time()
            pd.set_option('display.float_format', '{:.3f}'.format)

            # print("DEBUG: PAST DATA SENT TO ML PROCESS")
            # print(df["midpoint"])
            if strategy_dict["alpha"] == "informer":
                signals = self.informer.calculate_signals(df)
            if strategy_dict["alpha"] == "pyraformer":
                signals = self.pyraformer.calculate_signals(df)
            if strategy_dict["alpha"] == "transformer":
                signals = self.transformer.calculate_signals(df)

            # print("DEBUG: FUTURE PREDICTIONS")
            # print(signals["price"])

            # if trading decision making is based on heuristics, order generation is made on the server process
            if strategy_dict["trading_decision_making"] == "heuristics":
                signals['unix_timestamp'] = signals['Date'].apply(
                    lambda x: int(x.timestamp() * 1000))

                json_message = {
                    "type": "price_prediction",
                    "timestamp": float(df["timestamp"].iloc[-1]),
                    "timestamp_prediction": float(signals["unix_timestamp"].iloc[-1]),
                    "price_prediction": float(signals["price"].iloc[-1]),
                    "stop": 0,
                    "isDrl": 0}
                print("DEBUG: HEURISTICS RESPONSE")
                print(json_message)

                # send data back to server process
                serialized_data = pickle.dumps(json_message)
                # print("DEBUG: SENDING DATA BACK TO THE SERVER PROCESS FROM HERISTICS")
                conn.send(serialized_data)

            # if trading decision making is based on DRL, order generation is made here and simply executed on server_process
            if strategy_dict["trading_decision_making"] == "drl":

                past_data = df.values.tolist()
                prediction_data = signals.values.tolist()
                # Get only the last 2 columns from past_data = prices, spread
                past_data_ = [row[-2:] for row in past_data]
                # print(DEBUG: PAST DATA FOR DRL)
                # print(past_data_)
                # Get only the last column from prediction_data = prices
                prediction_data_ = [row[-1] for row in prediction_data]
                # print(DEBUG: PREDICTION DATA FOR DRL)
                # print(prediction_data_)
                action = self.drl.get_prediction(
                    past_data_, prediction_data_, strategy_dict)

                print("DEBUG: ACTION FROM DRL")
                print(action)

                order_direction = action[0].item()
                order_quantity = action[1]
                if order_direction == 0:
                    order_direction_ = "BUY"
                if order_direction == 1:
                    order_direction_ = "HOLD"
                if order_direction == 2:
                    order_direction_ = "SELL"
                signals['unix_timestamp'] = signals['Date'].apply(
                    lambda x: int(x.timestamp() * 1000))
                json_message = {
                    "type": "price_prediction",
                    "timestamp": float(df["timestamp"].iloc[-1]),
                    "timestamp_prediction": float(signals["unix_timestamp"].iloc[-1]),
                    "price_prediction": float(signals["price"].iloc[-1]),
                    "order_quantity": order_quantity,
                    "order_direction": order_direction_,
                    "stop": 0,
                    "isDrl": 1,
                }

                print("DRL RESPONSE")
                print(json_message)

                # send data back to server process
                serialized_data = pickle.dumps(json_message)
                # print("DEBUG: SENDING DATA BACK TO THE SERVER PROCESS")
                conn.send(serialized_data)

            # end_time = time.time()
            # execution_time = end_time - start_time
            # print(
            #    f"Execution time for calculate_signals: {execution_time:.4f} seconds")

        except Exception as e:
            print(f"Error in predict function: {e}")

    async def generate_predictions_and_orders_ml(self, conn):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            while True:
                try:
                    # get the number of active threads
                    num_threads = threading.active_count()
                    print(
                        f"Number of active threads in the ml process: {num_threads}")  # TODO how many threads should are predicted to be active?

                    serialized_data = conn.recv()

                    if serialized_data:
                        list_data_dict = pickle.loads(serialized_data)
                        strategy_dict = list_data_dict[-1]

                        if strategy_dict["stop"] == 1:
                            conn.send(pickle.dumps({"stop": 1}))

                        print("DEBUG: RECEIVED STRATEGY_DICT FROM SERVER PROCESS")
                        print(strategy_dict)

                        list_data_dict = list_data_dict[:-1]
                        df = pd.DataFrame(list_data_dict)

                        # Create a new thread to run each ml_prediction, this will allow us to run multiple ml_predictions at the same time
                        executor.submit(
                            self.predict, conn, df, strategy_dict)

                    else:
                        print("No data received")
                except EOFError:
                    print("EOFError encountered in simulate_predictions_and_trading()")

    def start(self, conn):
        print(f"ml_process running in process: {os.getpid()}")
        asyncio.run(self.generate_predictions_and_orders_ml(conn))


class ClientHandler:

    def __init__(self, websocket, conn, dataset):

        print("DEBUG: CREATING A NEW CLIENT HANDLER")

        self.websocket = websocket
        self.conn = conn
        self.is_connected = True

        self.dataset = dataset
        self.data_handler = None
        self.execution_handler = None
        self.portfolio = None

        self.response_times_s = []

        self.create_objects()

        # user defined parameters
        self.is_trade_running = False  # activated when the user clicks "trade"
        self.chosen_alpha = None
        self.chosen_trading_decision_making = None

    def create_objects(self):
        # read dataset choice
        if self.dataset == "sample-1":
            # file_path = "../data/processed/10-min-samples/sample-1-BTC.csv"
            file_path = "/app/data/processed/10-min-samples/sample-1-BTC.csv"
        elif self.dataset == "sample-2":
            # file_path = "../data/processed/10-min-samples/sample-2-BTC.csv"
            file_path = "/app/data/processed/10-min-samples/sample-2-BTC.csv"

        self.data_handler = HistoricCSVDataHandler(
            file_path)

        self.portfolio = Portfolio(
            self.data_handler)

        self.execution_handler = SimulatedExecutionHandler(
            self.data_handler)

    # Calculate response time and adding it to history of response times for this client
    def add_response_time(self, fill_timestamp, prediction_timestamp):
        response_time_ms = fill_timestamp - prediction_timestamp
        response_time_s = response_time_ms / 1000
        self.response_times_s.append(response_time_s)
        print("DEBUG: RESPONSE TIME (s): {:.3f}".format(response_time_s))

    async def run(self, websocket, conn):

        print("DEBUG: RUN() STARTS")
        # sequence length + label length # possible TODO allow for transformer variabiblity by reacting to chosen transformer args.txt/config/possible .yaml file
        lookback = 180 + 60
        reload = 30  # predictions every 30s, matching the time each transformer predicts into the future
        reload_count = reload

        dataset_length = len(self.data_handler.symbol_data)
        # print("DEBUG: LATEST BAR 1")
        # print(self.data_handler.get_latest_bar()["timestamp"])
        # print("DEBUG: FIRST SYMBOL")
        # print(self.data_handler.symbol_data.iloc[0]["timestamp"])
        for index, row in self.data_handler.symbol_data.iterrows():

            # print("DEBUG: TIME")
            # print(row["timestamp"])

            # update the latest symbol data
            self.data_handler.latest_symbol_data = self.data_handler.latest_symbol_data.append(
                row, ignore_index=True)
            # creates portfolio history based on timestamp
            self.portfolio.update_timeindex()

            # print("DEBUG: AFTER UPDATE ON TIMEINDEX")
            # print(row["timestamp"])
            # print(self.portfolio.all_holdings[-1])
            # print(self.portfolio.all_positions[-1])

            # as soon as "trade" is activated we start making predictions and trading
            if reload_count == reload and self.is_trade_running:
                reload_count = 0

                strategy_dict = {"portfolio_cash_balance": self.portfolio.current_holdings['cash'],
                                 "portfolio_current_stock": self.portfolio.current_positions['BTC'],
                                 "portfolio_max_stock": self.portfolio.max_stock,
                                 "portfolio_max_percentile_investment": self.portfolio.max_quantity_per_trade,
                                 "portfolio_initial_investment": self.portfolio.initial_capital,
                                 "alpha": self.chosen_alpha,
                                 "trading_decision_making": self.chosen_trading_decision_making,
                                 "stop": 0
                                 }

                data = self.data_handler.get_latest_bars(lookback)
                data_list_of_dicts = data.to_dict(orient='records')

                data_list_of_dicts.append(strategy_dict)
                serialized_data = pickle.dumps(data_list_of_dicts)
                conn.send(serialized_data)

            # send market update to client every 1 second
            json_message = {
                "type": "market_and_portofolio",
                "timestamp": row['timestamp'],
                "midpoint": row['midpoint'],
                "spread": row['spread'],
                "btc_holdings": self.portfolio.all_holdings[-1]["BTC"],
                "cash_holdings": self.portfolio.all_holdings[-1]["cash"],
                "commission_holdings": self.portfolio.all_holdings[-1]["commission"],
                "total_holdings": self.portfolio.all_holdings[-1]["total"],
                "btc_position": self.portfolio.all_positions[-1]["BTC"],
            }
            await websocket.send(json.dumps(json_message))

            if self.is_trade_running:
                reload_count += 1

            await asyncio.sleep(1)

            # Check if we have reached the end of the dataset
            if index == dataset_length - 1:
                print("Reached the end of the dataset")
                # Stop the trading process
                self.portfolio.create_equity_curve_dataframe()
                portfolio_stats = self.portfolio.output_summary_stats()
                # possible TODO execute evaluation also when a client disconnects or when a client stops trading
                print(portfolio_stats)
                print("Average Response Time from Prediction Initiation to Trade Execution: {:.3f}s".format(
                    np.mean(self.response_times_s)))
                print("Number of Trades Executed: {:.3f}".format(
                    len(self.response_times_s)))

                self.is_trade_running = False
                self.is_connected = False
                # Break out of the loop
                break

        print("DEBUG: RUN EXITING")

    async def trade_and_report(self, websocket, conn):

        while True and self.is_connected:
            try:
                # asynchronously receiving data from the connection conn by offloading the potentially blocking recv() call to a separate thread managed by the ThreadPoolExecutor, preventing the event loop from blocking.
                # even if the rate of recv() is faster than the rate of the calculations below, the order and the data will be correct
                # print("DEBUG: BEFORE RECV IN TRADE_AND_REPORT()")
                prediction = await websocket.loop.run_in_executor(None, lambda: conn.recv())
                # print("DEBUG: AFTER RECV IN TRADE_AND_REPORT()")

                if prediction:
                    prediction_dict = pickle.loads(prediction)

                    if prediction_dict["stop"] == 1:
                        # possible TODO clear tasks here??
                        break

                    # print("DEBUG: PREDICTION_DICT RECEIVED FROM ML PROCESS")
                    # print(prediction_dict)

                    order = None

                    if prediction_dict["isDrl"] == 1:

                        order = {
                            "symbol": "BTC",
                            "order_type": "MKT",
                            "quantity": prediction_dict["order_quantity"],
                            "direction": prediction_dict["order_direction"]
                        }

                        order = self.portfolio.check_order_feasibility(order)

                    if prediction_dict["isDrl"] == 0:
                        order = self.portfolio.generate_order(
                            prediction_dict)

                    if order != None:
                        fill_dict = await self.execution_handler.execute_order(order)
                        self.portfolio.update_fill(fill_dict)

                        # print("DEBUG: FILL")
                        # print(fill_dict)
                        # print("DEBUG: AFTER FILL:")
                        # print(self.portfolio.current_holdings)

                        self.add_response_time(
                            fill_dict["timestamp"], prediction_dict["timestamp"])

                        await websocket.send(json.dumps(prediction_dict))
                        await websocket.send(json.dumps(fill_dict))

                else:
                    print("No data received")

            except EOFError:
                print("EOFError encountered in trade_and_report()")
                break

            except Exception as e:
                print("An unexpected error occurred in trade_and_report():", e)
        print("DEBUG: TRADE_AND_REPORT EXITING")

    async def listen_client(self, websocket, conn):

        self.is_trade_running = False
        while True and self.is_connected:
            message = await websocket.recv()
            print(f"Received from client: {message}")

            try:
                message_json = json.loads(message)
                message_type = message_json.get("type")

                if message_type == "trade":
                    if not self.is_trade_running:
                        self.is_trade_running = True
                        self.chosen_alpha = message_json.get("alpha")
                        self.chosen_trading_decision_making = message_json.get(
                            "trading_decision_making")

                    else:
                        await websocket.send(json.dumps({"type": "error", "message": "Another trade command is currently running. Please wait."}))

                if message_type == "stop_trading":
                    if self.is_trade_running:
                        self.is_trade_running = False

                    else:
                        await websocket.send(json.dumps({"type": "error", "message": "No trade is running"}))

            except json.JSONDecodeError:
                print("Received message is not a JSON object")
                await websocket.send(json.dumps({"type": "error", "message": "Message is not a JSON object"}))

        print("DEBUG: LISTEN CLIENT EXITING ")

    async def start_(self):

        task1 = asyncio.create_task(self.run(self.websocket, self.conn))
        task2 = asyncio.create_task(
            self.listen_client(self.websocket, self.conn))
        task3 = asyncio.create_task(
            self.trade_and_report(self.websocket, self.conn))

        await asyncio.gather(task1, task2, task3)

    async def start(self):
        print(f"client handler running in process: {os.getpid()}")
        await self.start_()


class TradeAIServer:

    def __init__(self):
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        self.client_handler = None

    def clear_pipe(self):
        try:
            while self.parent_conn.poll():
                self.parent_conn.recv()
            while self.child_conn.poll():
                self.child_conn.recv()
        except Exception as e:
            print(f"Error occurred while clearing the pipe: {e}")

    def trade_process(self, conn):
        print(f"ml_process running in process: {os.getpid()}")

        ml_handler = MLHandler()
        ml_handler.start(conn)

    async def handle_connection(self, websocket, path, conn):

        print('Connection from', websocket.remote_address)

        initial_message = await websocket.recv()
        try:
            print(f"Received from client: {initial_message}")
            initial_message_json = json.loads(initial_message)

            if initial_message_json.get("type") != "connect":
                print("Invalid initial message from client")
                await websocket.send(json.dumps({"type": "error", "message": "Invalid initial message"}))
                if websocket.open:
                    await websocket.close()
                return

            dataset = initial_message_json.get("dataset")
            await websocket.send(json.dumps({"type": "log", "message": "Connected to server"}))
            self.client_handler = ClientHandler(websocket, conn, dataset)
            await self.client_handler.start()

        except json.JSONDecodeError:
            print("Initial message from client is not a JSON object")
            await websocket.send(json.dumps({"type": "error", "message": "Initial message is not a JSON object"}))

        except websockets.exceptions.ConnectionClosedOK:
            print(f"Client {websocket.remote_address} disconnected")

        except websockets.exceptions.ConnectionClosed as e:
            print(
                f"Connection with {websocket.remote_address} closed unexpectedly: {e}")

        except Exception as e:
            print(f"Unexpected error occurred: {e}")
            await websocket.send(json.dumps({"type": "error", "message": "Unexpected error occurred"}))

        finally:
            if websocket.open:
                await websocket.close()

            self.child_conn.send(pickle.dumps(
                {"stop": 1}))  # avoids broken pipe
            # possible TODO: self.client_handler = None ??  # self.clear_pipe()
            # possible TODO calculate stats if client disconnects?
            print(f"Connection with {websocket.remote_address} closed.")

    def server_process(self, conn):
        print(f"server_process running in process: {os.getpid()}")

        async def main(self):
            async with websockets.serve(lambda websocket, path: self.handle_connection(websocket, path, self.parent_conn), "0.0.0.0", 80, ping_interval=1000, ping_timeout=1000):
                print("Server started. Listening at ws://0.0.0.0:80")
                await asyncio.Future()

        asyncio.run(main(self))

    def start(self):
        print(f"number of processors: {os.cpu_count()}")
        print(f"main running in process: {os.getpid()}")

        # 2 new processes
        p1 = multiprocessing.Process(
            target=self.server_process, args=(self.parent_conn,))
        p2 = multiprocessing.Process(
            target=self.trade_process, args=(self.child_conn,))

        p1.start()
        p2.start()

        p1.join()
        p2.join()

    def remove_client_handler(self, client_handler):
        if client_handler in self.client_handlers:
            self.client_handlers.remove(client_handler)


if __name__ == '__main__':
    tradeai_server = TradeAIServer()
    tradeai_server.start()
