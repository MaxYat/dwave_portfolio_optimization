from qiskit_finance import QiskitFinanceError
from qiskit_finance.data_providers import *
import numpy as np


def load_stock_market_data(tickers, start, end):
    result_good_tickers = []
    result_good_data = []
    try:
        data = YahooDataProvider(tickers=tickers, start=start, end=end)
        data.run()

        for (i, ticker) in enumerate(data._tickers):
            prices = data._data[i].to_numpy()
            if np.any(np.isnan(prices)):
                continue
            result_good_data.append(prices)
            result_good_tickers.append(ticker)

    except QiskitFinanceError as ex:
        print(ex)

    return np.asarray(result_good_data), result_good_tickers