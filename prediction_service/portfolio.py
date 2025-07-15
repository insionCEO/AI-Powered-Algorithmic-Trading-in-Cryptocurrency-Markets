#!/usr/bin/python
# -*- coding: utf-8 -*-

# portfolio.py

from __future__ import print_function

import datetime
from math import floor
import numpy as np
import pandas as pd
# from performance import create_sharpe_ratio, create_drawdowns
from data import HistoricCSVDataHandler


class Portfolio(object):

    def __init__(self, data_handler, initial_capital=100000.0):
        # self.bars = bars
        # self.events = events
        # self.symbol_list = self.bars.symbol_list  # to comment out everywhere

        self.equity_curve = None
        self.total_time_diff_s = None

        self.initial_capital = initial_capital

        # self.start_date = start_date
        # self.latest_market_data = None
        self.data_handler = data_handler

        # for DRL and risk management
        # max stock to hold for bitcoin
        self.max_stock = 1
        # per trade and relative to initial capital
        self.max_quantity_per_trade = 0.01

        self.all_positions = self.construct_all_positions()
        # self.current_positions = dict(
        #    (k, v) for k, v in [(s, 0) for s in self.symbol_list])
        self.current_positions = {'BTC': 0.0}

        self.all_holdings = self.construct_all_holdings()
        self.current_holdings = self.construct_current_holdings()

    # positions are just the stocks we have in our portfolio
    # "all_positions" means positions tracked over time
    def construct_all_positions(self):

        d = {
            'timestamp':  self.data_handler.latest_symbol_data.iloc[-1]['timestamp'], 'BTC': 0.0}
        return [d]

    # holdings are the stocks we have in our portfolio + cash + what was spent on commisions
    def construct_all_holdings(self):

        d = {
            'timestamp': self.data_handler.latest_symbol_data.iloc[-1]['timestamp'],
            'BTC': 0.0,
            'cash': self.initial_capital,
            'commission': 0.0,
            'total': self.initial_capital}
        return [d]

    def construct_current_holdings(self):

        d = {'BTC': 0.0,
             'cash': self.initial_capital,
             'commission': 0.0,
             'total': self.initial_capital}
        return d

    # Update time in portfolio
    def update_timeindex(self):

        # print("MIDPOINT FROM PORTFOLIO: " +
        #      str(self.data_handler.latest_symbol_data.iloc[-1]['midpoint']) + " at " + str(self.data_handler.latest_symbol_data.iloc[-1]['timestamp']))

        latest_timestamp = self.data_handler.latest_symbol_data.iloc[-1]['timestamp']
        latest_midpoint = self.data_handler.latest_symbol_data.iloc[-1]['midpoint']

        symbol = 'BTC'

        # Update positions
        # ================
        dp = {'timestamp': latest_timestamp, 'BTC': 0.0}
        dp['BTC'] = self.current_positions['BTC']

        # Append the current positions
        self.all_positions.append(dp)

        # Update holdings
        # ===============
        dh = {
            'timestamp': latest_timestamp,
            'BTC': 0.0,
            'cash': self.current_holdings['cash'],
            'commission': self.current_holdings['commission'],
            'total': self.current_holdings['cash']}

        # Approximation to the real value
        market_value = self.current_positions['BTC'] * \
            latest_midpoint
        dh['BTC'] = market_value
        dh['total'] += market_value

        # Append the current holdings
        self.all_holdings.append(dh)

        # print(latest_midpoint)
        # print(self.current_positions['BTC'])
        # print("UPDATED HOLDINGS ON TIMESTAMP: ")
        # print(self.all_holdings[-1])

    # ======================
    # FILL/POSITION HANDLING
    # ======================

    # Check if we can execute the order
    def check_order_feasibility(self, order):

        latest_midpoint = self.data_handler.latest_symbol_data.iloc[-1]['midpoint']
        # print("DEBUG: Latest midpoint:", latest_midpoint)

        # based on IB commission
        # commission = max(1.3, 0.013 * order["quantity"])
        # based on Binance exchange commission under 1000000 BUSD monthly volume
        commission = 0.0001 * (latest_midpoint*order["quantity"])
        # print("DEBUG: Commission:", commission)

        # check if we have enough cash to buy
        if order['direction'] == 'BUY' and self.current_holdings['cash'] < (order["quantity"] * latest_midpoint + commission):
            # print("DEBUG: Not enough cash to buy.")
            order = None

        # check if we have enough stock to sell
        elif order['direction'] == 'SELL' and self.current_positions['BTC'] <= 0:
            # print("DEBUG: Not enough stock to sell.")
            order = None

        # don't do anything if we want to hold
        elif order['direction'] == 'HOLD':
            # print("DEBUG: Hold direction.")
            order = None

        elif order['quantity'] <= 0:
            # print("DEBUG: Invalid quantity.")
            order = None

        elif order['quantity'] + self.current_positions['BTC'] > self.max_stock:
            # print("DEBUG: Max stock reached.")
            order = None

        return order

    # Heuristics-based order generation
    def generate_order(self, signal):

        # NOTE this is 30 seconds into the future
        # TODO use threshold based or moving average crossover strategy

        order = None

        # fixed quantity
        quantity = self.max_quantity_per_trade

        latest_timestamp = self.data_handler.latest_symbol_data.iloc[-1]['timestamp']
        latest_midpoint = self.data_handler.latest_symbol_data.iloc[-1]['midpoint']
        price_prediction_timestamp = signal["timestamp_prediction"]
        price_prediction = signal["price_prediction"]

        future_midpoint_change = price_prediction - latest_midpoint

        # print("DEBUG: FUTURE MIDPOINT CHANGE: ")
        # print(future_midpoint_change)

        # Calculate slippage
        # This calculation assumes that a 250 ms delay in order execution results in a price change equal to 0.1% (0.001) of the latest midpoint price.
        # The latest midpoint price is the average of the bid and ask prices.

        # latency_slippage = 0.001 * latest_midpoint  # 250 ms slippage

        # Net profit without commission
        net_profit = future_midpoint_change  # - latency_slippage

        # Calculate the commission
        # based on Binance exchange commission under 1000000 BUSD monthly volume
        commission = 0.0001 * (latest_midpoint*quantity)
        # based on IB commission
        # commission = max(1.3, 0.013 * quantity)
        # print("DEBUG: COMMISSION: ")
        # print(commission)

        T = 0.5
        # Check if net profit after commission is positive
        if net_profit - commission > T:
            direction = "BUY"
        elif net_profit - commission < -T:
            direction = "SELL"
            quantity = self.current_positions['BTC']
        else:
            direction = "HOLD"

        print("DEBUG: ORDER DIRECTION: ")
        print(direction)

        order = {
            "symbol": "BTC",
            "order_type": "MKT",
            "quantity": quantity,
            "direction": direction
        }

        order = self.check_order_feasibility(order)

        return order

    def update_positions_from_fill(self, fill):
        fill_dir = 0
        if fill["direction"] == 'BUY':
            fill_dir = 1
        if fill["direction"] == 'SELL':
            fill_dir = -1

        # Update positions list with new quantities
        self.current_positions[fill["symbol"]] += fill_dir*fill["quantity"]

    def update_holdings_from_fill(self, fill):
        # Check whether the fill is a buy or sell
        fill_dir = 0
        if fill["direction"] == 'BUY':
            fill_dir = 1
        if fill["direction"] == 'SELL':
            fill_dir = -1

        cost = fill_dir * fill["fill_cost"]
        self.current_holdings[fill["symbol"]] += cost
        self.current_holdings['commission'] += fill["commission"]
        self.current_holdings['cash'] -= (cost + fill["commission"])
        self.current_holdings['total'] -= (cost + fill["commission"])

    # receive fill after order execution and update portfolio accordingly
    def update_fill(self, fill):
        self.update_positions_from_fill(fill)
        self.update_holdings_from_fill(fill)

    # ====================
    # POST BACKTEST STATS
    # ====================

    # Calculates annualized sharpe Ratio, based on a custom period of returns in seconds
    # Create the Sharpe ratio for the strategy, based on a benchmark of zero (i.e. no risk-free rate information).

    def create_sharpe_ratio(self, returns, return_length_seconds):

        trading_days_per_year = 365  # bitcoin trades 24/7
        seconds_per_trading_day = 6.5 * 60 * 60
        total_seconds_per_year = trading_days_per_year * seconds_per_trading_day
        periods = return_length_seconds / total_seconds_per_year
        return np.sqrt(periods) * (np.mean(returns)) / np.std(returns)

    import numpy as np
    import pandas as pd

    # Calculate the drawdown, maximum drawdown, and drawdown duration.
    def create_drawdowns(self, pnl):
        # Calculate the drawdown series
        # Dividing the profit and loss series by its cumulative maximum and then subtracting 1 gives the drawdown percentage at each point in time.
        drawdown = pnl / pnl.cummax() - 1.0

        # Calculate the maximum drawdown
        max_drawdown = drawdown.min()

        # Calculate the drawdown duration
        drawdown_duration = 0
        max_duration = 0
        for value in drawdown:
            if value == 0:
                drawdown_duration = 0
            else:
                drawdown_duration += 1
                max_duration = max(max_duration, drawdown_duration)

        return drawdown, max_drawdown, max_duration

    def output_summary_stats(self):
        total_return = self.equity_curve['equity_curve'].iloc[-1]
        # print("DEBUG: Total Return:", total_return)

        returns = self.equity_curve['returns']
        # print("DEBUG: Returns:", returns)

        pnl = self.equity_curve['equity_curve']  # profit_and_loss
        # print("DEBUG: Profit and Loss:", pnl)

        sharpe_ratio = self.create_sharpe_ratio(
            returns, self.total_time_diff_s)
        # print("DEBUG: Sharpe Ratio:", sharpe_ratio)

        drawdown, max_dd, max_dd_duration = self.create_drawdowns(pnl)
        # print("DEBUG: Drawdown", drawdown, "Max Drawdown:",
        #      max_dd, "Max Drawdown Duration:", max_dd_duration)

        self.equity_curve['drawdown'] = drawdown
        # print("DEBUG: Equity curve with drawdown:", self.equity_curve)

        stats = [("Total Net Return Percentage", "%0.6f%%" % ((total_return - 1.0) * 100.0)),  # subtracting 1 (corresponds to initial investment) and mutliplying by 100 (getting percentage)
                 ("Sharpe Ratio", "%0.6f" % sharpe_ratio),
                 ("Max Drawdown", "%0.6f%%" % (max_dd * 100.0)),
                 ("Max Drawdown Duration", "%ds" % max_dd_duration),
                 ("Effective Execution Time", "%ds" % self.total_time_diff_s)]

        return stats

    def create_equity_curve_dataframe(self):
        pd.set_option("display.max_rows", None)

        # print("DEBUG: ALL HOLDINGS")
        # print(self.all_holdings)
        # print("DEBUG: EXEC TIME")
        # getting the correct time execution time in seconds to calculate sharpe ratio
        time_diff_ms = self.all_holdings[-1]['timestamp'] - \
            self.all_holdings[0]['timestamp']
        # millisseconds to seconds, rounded down
        self.total_time_diff_s = int(time_diff_ms // 1000)
        print("EXEC TIME")
        print(self.total_time_diff_s)

        curve = pd.DataFrame(self.all_holdings)
        curve.set_index('timestamp', inplace=True)
        # percentage change betweeent the current and prior element
        curve['returns'] = curve['total'].pct_change()
        curve['equity_curve'] = (1.0+curve['returns']).cumprod()
        self.equity_curve = curve
