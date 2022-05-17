import numpy as np

def norm_prices(prices_history):
    price = prices_history[:, -1]
    return prices_history / price[:, None]