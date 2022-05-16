from qiskit_finance import QiskitFinanceError
from qiskit_finance.data_providers import *
import datetime
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

# from https://qiskit.org/documentation/finance/tutorials/11_time_series.html

try:
    data = YahooDataProvider(
        tickers=["AEO", "ABBY", "AEP", "AAL", "AFN"],
        start=datetime.datetime(2018, 1, 1),
        end=datetime.datetime(2018, 12, 31),
    )
    data.run()
    for (cnt, s) in enumerate(data._tickers):
        plt.plot(data._data[cnt], label=s)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3)
    plt.xticks(rotation=90)
    plt.show()
except QiskitFinanceError as ex:
    data = None
    print(ex)