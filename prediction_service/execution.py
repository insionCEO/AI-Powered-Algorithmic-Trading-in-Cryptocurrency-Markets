#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
from abc import ABCMeta, abstractmethod
import datetime
from data import HistoricCSVDataHandler
import asyncio


class SimulatedExecutionHandler(object):

    def __init__(self, data_handler):
        self.data_handler = data_handler

    # converts Order objects into Fill objects
    async def execute_order(self, order):

        # NOTE the fill object already represents the trade executed
        # NOTE not filling fill ratio

        # artificially adding latency to simulate slippage
        await asyncio.sleep(0.25)

        # print("DEBUG: MIDPOINT FROM ORDER EXECUTION: " +
        #      str(self.data_handler.latest_symbol_data.iloc[-1]['midpoint']) + " at " + str(self.data_handler.latest_symbol_data.iloc[-1]['timestamp']))
        latest_timestamp = self.data_handler.latest_symbol_data.iloc[-1]['timestamp']
        latest_midpoint = self.data_handler.latest_symbol_data.iloc[-1]['midpoint']

        fill_cost = order["quantity"] * latest_midpoint
        # based on IB commission
        # commission = max(1.3, 0.013 * order["quantity"])
        # based on Binance exchange commission under 1000000 BUSD monthly volume
        commission = 0.0001 * (latest_midpoint*order["quantity"])

        fill = {
            "type": "fill",
            "timestamp": latest_timestamp,
            "traded_price": latest_midpoint,
            "symbol": order["symbol"],
            "quantity": order["quantity"],
            "direction": order["direction"],
            "order_type": order["order_type"],
            "fill_cost": fill_cost,
            "commission": commission
        }

        return fill
