import pandas as pd
import numpy as np

def compute_mid_price(df):
    return (df['ask_price_1'] + df['bid_price_1']) / 2

def compute_spread(df):
    return df['ask_price_1'] - df['bid_price_1']

def compute_order_flow_imbalance(df):
    return (df['bid_size_1'] - df['ask_size_1']) / (df['bid_size_1'] + df['ask_size_1'] + 1e-6)

def compute_queue_imbalance(df):
    return df['bid_size_1'] / (df['bid_size_1'] + df['ask_size_1'] + 1e-6)

def compute_price_momentum(df, window=10):
    mid_price = compute_mid_price(df)
    return mid_price.diff(periods=window)

def add_all_features(df):
    df = df.copy()
    df['mid_price'] = compute_mid_price(df)
    df['spread'] = compute_spread(df)
    df['ofi'] = compute_order_flow_imbalance(df)
    df['qi'] = compute_queue_imbalance(df)
    df['momentum'] = compute_price_momentum(df)
    return df