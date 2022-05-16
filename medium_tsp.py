import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, permutations
from math import sqrt, pi, sin, cos

import networkx as nx
import dwave_networkx as dnx
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dimod.reference.samplers import ExactSolver
from dwave.cloud.client import Client

# Import the problem inspector to begin data capture
import dwave.inspector

num_cities = 11
jump = 2
city_list = np.array([[cos(2*pi*i*jump/num_cities), sin(2*pi*i*jump/num_cities)] for i in range(num_cities)])


def distance(i, j):
    return sqrt((city_list[i][0] - city_list[j][0])**2 + (city_list[i][1] - city_list[j][1])**2)


def show_solution(solution, title):
    city_locations = np.array([city_list[solution[i]] for i in range(len(solution))])

    plt.scatter(city_list[:, 0], city_list[:, 1])
    plt.plot(city_locations[:, 0], city_locations[:, 1])
    plt.title(title)
    plt.show()


def nodes_to_solution(nodes):
    n = round(sqrt(len(nodes)))
    solution = np.zeros(shape=(n + 2), dtype=np.int16)

    ones = 0
    cols = rows = set()
    for i in range(len(nodes)):
        if nodes[i] != 1:
            continue
        ones += 1
        row = i // n
        col = i % n
        cols.add(col)
        rows.add(row)
        solution[row + 1] = col + 1

    status = ones == len(cols) == len(rows) == n

    return solution, status


initial_solution = np.arange(city_list.shape[0] + 1)
initial_solution[-1] = 0
show_solution(initial_solution, title="Initial solution")

n = len(city_list)-1
# nodes = np.zeros((n,n), dtype=np.float32)

edges = []

# horizontal edges
for i in range(n):
    edges.extend(list(combinations(range(i*n, i*n + n),2)))

# edges between layers
for i in range(n-1):
    edges.extend([(a,b+n) for a,b in permutations(range(i*n, i*n + n),2)])

# vertical edges
for i in range(n):
    edges.extend([(a*n+i, b*n+i) for a,b in combinations(range(n),2)])

G = nx.Graph()
G.add_edges_from(edges)

node_bonus = 10
hard_constraint_hor_penalty = 30
hard_constraint_ver_penalty = 30

Q = {(node, node): distance(0, node + 1) - node_bonus for node in range(n)}

Q.update({(node, node): - node_bonus for node in range(n, len(G.nodes)-n)})

Q.update({(node, node): distance(node % n + 1, 0) - node_bonus for node in range(len(G.nodes)-n, len(G.nodes))})


Q.update({(i, j): distance(i % n + 1, j % n + 1) for i, j in G.edges if i // n != j // n and i % n != j % n})

Q.update({(i, j): hard_constraint_hor_penalty for i, j in G.edges if i // n == j // n})
Q.update({(i, j): hard_constraint_ver_penalty for i, j in G.edges if i % n == j % n})

# sampler = ExactSolver()
# S = dnx.maximum_independent_set(G, sampler=sampler)

# sampler = EmbeddingComposite(DWaveSampler())
# S = dnx.maximum_independent_set(G, sampler=sampler, num_reads=10, label='TSP')

# print('Maximum independent set size found is', len(S))
# print(S)

if num_cities <= 5:
    exact_sampler = ExactSolver()
    exact_samples = exact_sampler.sample_qubo(Q)
    exact_nodes = exact_samples.lowest().record[0][0]

    exact_solution, exact_status = nodes_to_solution(exact_nodes)
    print(f"exact_status={exact_status}")
    print(f"exact_solution={exact_solution}")
    print(f"Solution energy={exact_samples.first.energy}")
    show_solution(exact_solution, title="Solution of ExactSolver")

token = "DEV-49e25465ebda65be7ab0864b9276b89d8b98fd94"

client = Client(token=token)
print(client.get_solvers())

dwave_sampler = EmbeddingComposite(DWaveSampler(token=token))
dwave_samples = dwave_sampler.sample_qubo(Q, num_reads=10000, chain_strength=100, label=f'Study TSP ({num_cities})')
dwave_sample = dwave_samples.lowest()
dwave_nodes = dwave_sample.record[0][0]

dwave_solution, dwave_status = nodes_to_solution(dwave_nodes)
print(f"dwave_status={dwave_status}")
print(f"dwave_solution={dwave_solution}")
print(f"Solution energy={dwave_samples.first.energy}")
print(f"Chain break fraction={dwave_samples.first.chain_break_fraction}")

print(f"dwave_samples.info={dwave_samples.info}")
show_solution(dwave_solution, title="Solution of DwaveSolver")

dwave.inspector.show(dwave_samples)

