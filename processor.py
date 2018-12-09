'''
This file is used to process the raw data & generate features.

Reference: https://github.com/philipperemy/deep-learning-bitcoin
'''

import math
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from constants import *

class Processor:

    def __init__(self):
        # suppose alway put the .csv data file in the 'data' fold in the same directory as this file
        self.dataset_path = "data/bitcoin.csv"
        self.preprocess()

    @property 
    def UTC_time(self):
        return self._UTC_time

    @property 
    def price(self):
        return self._price

    @property 
    def var(self):
        return self._var

    @property 
    def reddit_doc(self):
        return self._reddit_doc

    @property 
    def MACD(self):
        return self._MACD
        
    # the data imported has no holes & is average from 3 exchanges    
    def preprocess(self):
        self._data = pd.read_csv(self.dataset_path)
        print('Columns found in the dataset {}'.format(self._data.columns))
        self._data = self._data.dropna() # drop the rows containing NaN
        # bitcoin price averaged from 3 different exchanges
        self._price = self._data['Average_price'].values

        # size of the whole data set
        total_num_data = len(self._price)
        train_start = math.floor(0.7 * total_num_data)

        self._price = self._price[train_start:]
        self.num_data = len(self._price)
        print('{} data points in total.'.format(self.num_data))

        # the count of reddit documents that countain the word 'hack'
        self._reddit_doc = self._data['Reddit_count'].values
        self._reddit_doc = self._reddit_doc[train_start:]

        # MACD
        self._MACD = self._data['MACD'].values
        self._MACD = self._MACD[train_start:]

        # variation
        self._var = self._data['Average_high'].values - self._data['Average_low'].values
        self._var = self._var[train_start:]

        timestamps = self._data['Timestamp'].values
        # convert the timstamps to UTC: Coordinated Universal Time (same as GMT: Greenwich Mean Time)
        self._UTC_time = []
        for timestamp in timestamps:
            utc_time = datetime.utcfromtimestamp(timestamp)
            self._UTC_time.append(utc_time.strftime("%Y-%m-%d %H:%M:%S.%f+00:00 (UTC)"))
        self._UTC_time = np.array(self._UTC_time)
        self._UTC_time = self.UTC_time[train_start:]  

        self._data = None # free memory   
    
    def get_channels(self, current):
        return self._price[current]-self._price[current-1], self._reddit_doc[current], self._MACD[current] #,self._var[current]