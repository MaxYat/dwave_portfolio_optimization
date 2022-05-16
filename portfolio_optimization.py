import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, permutations
from math import sqrt, pi, sin, cos
from datetime import datetime

from load_stock_market_data import load_stock_market_data

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
BUDGET = 1000
LAGRANGE = 100

assets = ['AEO', 'ABBY', 'AEP', 'AAL']
start = datetime(2018, 1, 1)
end = datetime(2018, 12, 31)

AXIS_ASSET, AXIS_TIME = 0, 1
prices_history, good_tickers = load_stock_market_data(tickers=assets, start=start, end=end)

cov = np.cov(prices_history)

expected_price = np.average(prices_history, axis=AXIS_TIME)

price = prices_history[:, -1]

expected_profit = expected_price - price


def check_status(solution):
    budget = np.sum(price * solution)
    return budget == BUDGET


def nodes_to_solution(nodes):
    solution = np.zeros(len(assets), dtype=float)
    for i in range(len(nodes)):
        solution[i // BITS_PER_ASSET] += nodes[i] * 2**(i % BITS_PER_ASSET)
    return solution, check_status(solution)


def get_metrices(solution):
    expected_return = np.sum(solution * (expected_price - price))

    risk = 0.5 * solution.T @ cov @ solution

    sharpe_ratio = expected_return / np.sqrt(2 * risk)

    return expected_return, risk, sharpe_ratio


def show_solution(solution):
    print(f"Asset | price per unit | units | price | percentage of budget")
    total = 0
    for i, asset in enumerate(assets):
        asset_price = solution[i] * price [i]
        total += asset_price
        print(f"{asset} | {price[i]:.2f} | {solution[i]} | {asset_price:.2f} | {asset_price / BUDGET * 100 : .2f} %")
    print(f"Total spent: {total:.2f}")
    print(f"Budget fulfilled: {total / BUDGET * 100:.2f} %")


# QUBO graph with bit representation of quantities of selected assets
Q = {(node, node):
          - expected_profit[node // BITS_PER_ASSET] * 2 ** (node % BITS_PER_ASSET) # profit
     + LAGRANGE * (price[node // BITS_PER_ASSET] * 2 ** (2 * (node % BITS_PER_ASSET)) # budget
     - 2 * BUDGET * 2 ** (node % BITS_PER_ASSET))
     for node in range(len(assets) * BITS_PER_ASSET)}

Q.update({(node1, node2):
               RISK_APPETITE * cov[node1 // BITS_PER_ASSET, node2 // BITS_PER_ASSET] *
               2**(node1 % BITS_PER_ASSET + node2 % BITS_PER_ASSET)
          for node1, node2 in combinations(range(len(assets) * BITS_PER_ASSET), 2)
          if (node1 // BITS_PER_ASSET < node2 // BITS_PER_ASSET)
          })


exact_sampler = ExactSolver()
exact_samples = exact_sampler.sample_qubo(Q)
exact_nodes = exact_samples.lowest().record[0][0]

exact_solution, exact_status = nodes_to_solution(exact_nodes)

show_solution(exact_solution)

expected_return, risk, sharpe_ratio = get_metrices(exact_solution)

print(f"Expected return = {expected_return:.2f}")
print(f"Risk = {risk:.2f}")
print(f"Sharpe ratio = {sharpe_ratio:.2f}")
