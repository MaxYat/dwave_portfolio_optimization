import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, permutations
from math import sqrt, pi, sin, cos
from datetime import datetime

from load_stock_market_data import load_stock_market_data
from utils_financial import norm_prices

import networkx as nx
import dwave_networkx as dnx
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dimod.reference.samplers import ExactSolver
from dwave.cloud.client import Client

# Import the problem inspector to begin data capture
# import dwave.inspector

BITS_PER_ASSET = 5
RISK_APPETITE = 0.1
LAGRANGE = 100

# now it's one dollar )
# BUDGET = 1

assets = ['AEO', 'ABBY', 'AEP', 'AAL']
start = datetime(2018, 1, 1)
end = datetime(2018, 12, 31)

AXIS_ASSET, AXIS_TIME = 0, 1
prices_history, good_tickers = load_stock_market_data(tickers=assets, start=start, end=end)

# After norm, prices history array will contain values how much
# one had have to invest into each asset to get 1 dollar in return.
prices_history = norm_prices(prices_history)

cov = np.cov(prices_history)

# expected return per one dollar investment
expected_return_per_dollar = np.average(prices_history, axis=AXIS_TIME)

# cov = np.zeros_like(cov)
# expected_return_per_dollar = np.zeros_like(expected_return_per_dollar)


# after norm it's all ones
# price = prices_history[:, -1]

# expected_profit = expected_price - price


def check_status(solution):
    return np.sum(solution) == 1


def nodes_to_solution(nodes):
    solution = np.zeros(len(assets), dtype=float)
    for i in range(len(nodes)):
        solution[i // BITS_PER_ASSET] += nodes[i] * 2**(i % BITS_PER_ASSET)

    solution /= 2 ** BITS_PER_ASSET - 1

    return solution, check_status(solution)


def get_metrices(solution):
    expected_return = np.sum(solution * expected_return_per_dollar)

    risk = 0.5 * solution.T @ cov @ solution

    sharpe_ratio = expected_return / np.sqrt(2 * risk)

    return expected_return, risk, sharpe_ratio


def show_solution(solution):
    print(f"Asset | expected_return_per_dollar | part of investment budget")
    total = 0
    for i, part in enumerate(solution):
        total += part
        print(f"{assets[i]} | {expected_return_per_dollar[i]:.2f} | {part} ")
    print(f"Total spent: {total:.2f}")

    expected_return, risk, sharpe_ratio = get_metrices(solution)

    print(f"Expected return = {expected_return:.2f}")
    print(f"Risk = {risk:.2f}")
    print(f"Sharpe ratio = {sharpe_ratio:.2f}")


# QUBO graph with bit representation of quantities of selected assets

budget = 2 ** BITS_PER_ASSET - 1

Q = {(node, node):
         - expected_return_per_dollar[node // BITS_PER_ASSET] * 2 ** (node % BITS_PER_ASSET)  # profit
         + LAGRANGE * (2 ** (2 * (node % BITS_PER_ASSET)) - 2 * budget * 2 ** (node % BITS_PER_ASSET)) # budget
     for node in range(len(assets) * BITS_PER_ASSET)}

Q.update({(node1, node2):
              ((RISK_APPETITE * cov[node1 // BITS_PER_ASSET, node2 // BITS_PER_ASSET] *
               2**(node1 % BITS_PER_ASSET + node2 % BITS_PER_ASSET))
               if (node1 // BITS_PER_ASSET < node2 // BITS_PER_ASSET) else 0) + # risk
              LAGRANGE * 2 * 2**(node1 % BITS_PER_ASSET + node2 % BITS_PER_ASSET) # budget
          for node1, node2 in combinations(range(len(assets) * BITS_PER_ASSET), 2)
          })


exact_sampler = ExactSolver()
exact_samples = exact_sampler.sample_qubo(Q)
exact_nodes = exact_samples.lowest().record[0][0]

exact_solution, exact_status = nodes_to_solution(exact_nodes)

show_solution(exact_solution)

