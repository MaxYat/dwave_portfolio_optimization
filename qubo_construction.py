from sympy.abc import i, N, b, B, k
from sympy import Sum, factorial, oo, IndexedBase, Function

w = IndexedBase('w')
budget_constraint = (Sum(w[i], (i, 0, N-1)) - k)**2
print(budget_constraint.expand())

bit_budget = (Sum(Sum(2**b * w[i,b], (b, 0, B-1)), (i, 0, N-1)) - k)**2
print(bit_budget.expand())