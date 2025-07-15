#!/usr/bin/python
# -*- coding: utf-8 -*-


from __future__ import print_function

from abc import ABCMeta, abstractmethod
import datetime
import os
import os.path

import numpy as np
import pandas as pd


class HistoricCSVDataHandler(object):

    def __init__(self, csv_dir):
        self.csv_dir = csv_dir

        self.symbol_data = None
        self.latest_symbol_data = None
        self.continue_backtest = True
        self.bar_index = 0

        self._open_convert_csv_files()

    def _open_convert_csv_files(self):

        self.symbol_data = pd.read_csv(self.csv_dir)
        self.latest_symbol_data = self.symbol_data.iloc[:240].reset_index(
            drop=True)  # Initialize as a DataFrame instead of a list
        self.symbol_data = self.symbol_data.iloc[240:].reset_index(drop=True)

    def get_latest_bar(self):
        """
        Returns the last bar from the latest_symbol list.
        """
        return self.latest_symbol_data.iloc[-1]

    def get_latest_bars(self, N=1):
        """
        Returns the last N rows from the latest_symbol_data DataFrame.
        """
        return self.latest_symbol_data.iloc[-N:]
