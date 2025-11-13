def generate_labels(df):
    df = df.copy()
    df['mid_price'] = (df['ask_price_1'] + df['bid_price_1']) / 2
    df['future_mid'] = df['mid_price'].shift(-1)
    df['label'] = (df['future_mid'] > df['mid_price']).astype(int)
    return df