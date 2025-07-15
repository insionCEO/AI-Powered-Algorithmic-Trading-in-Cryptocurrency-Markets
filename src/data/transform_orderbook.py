import pandas as pd
import math as math
import csv
df = pd.read_csv("./binance_data/btc_update.csv")

def average(a, b):
    result = (a+b)/2
    return result

def calc_spread(a, b):
    return abs(a-b)

timestamps = df['timestamp']

timestamps = list(dict.fromkeys(timestamps))

filt1 = df['timestamp'] == timestamps[0]
filt2 = df['side'] == 'a'
filt3 = df['side'] == 'b'
filt4 = df['qty'] > 0

headers = ['timestamp', 'lowest_ask', 'highest_bid', 'midpoint', 'spread']

f = open('binance_data/transformed.csv', 'w')

writer = csv.writer(f)

writer.writerow(headers)

for step in timestamps:
    filt1 = df['timestamp'] == step
    lowest_ask = df.loc[filt1 & filt2 & filt4]['price'].min() 
    highest_bid = df.loc[filt1 & filt3 & filt4]['price'].max()
    if (math.isnan(lowest_ask) or math.isnan(highest_bid)):
        continue

    midpoint = average(lowest_ask, highest_bid)
    spread = calc_spread(lowest_ask, highest_bid)

    row = [step, lowest_ask, highest_bid, midpoint, spread]
    writer.writerow(row)

    