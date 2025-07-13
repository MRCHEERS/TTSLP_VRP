import random
import copy
from typing import List, Tuple

from .operators import random_removal, worst_distance_removal, two_opt, swap_between_routes


class ALNSSolver:
    """Simple ALNS solver for CVRP style problems."""

    def __init__(self, dist_matrix, demands, capacity, depot=0):
        self.dist = dist_matrix
        self.demands = demands
        self.capacity = capacity
        self.depot = depot
        self.customers = [i for i in range(len(demands)) if i != depot]

    def initial_solution(self) -> List[List[int]]:
        routes = []
        cur_route = [self.depot]
        load = 0
        for c in self.customers:
            d = self.demands[c]
            if load + d > self.capacity:
                cur_route.append(self.depot)
                routes.append(cur_route)
                cur_route = [self.depot, c]
                load = d
            else:
                cur_route.append(c)
                load += d
        cur_route.append(self.depot)
        routes.append(cur_route)
        return routes

    def route_cost(self, route: List[int]) -> float:
        cost = 0.0
        for i in range(len(route) - 1):
            cost += self.dist[route[i]][route[i + 1]]
        return cost

    def evaluate(self, solution: List[List[int]]) -> float:
        if not self.is_feasible(solution):
            return 1e9
        return sum(self.route_cost(r) for r in solution)

    def is_feasible(self, solution: List[List[int]]) -> bool:
        visited = []
        for route in solution:
            load = 0
            if route[0] != self.depot or route[-1] != self.depot:
                return False
            for n in route[1:-1]:
                load += self.demands[n]
                visited.append(n)
            if load > self.capacity:
                return False
        return sorted(visited) == sorted(self.customers)

    def greedy_insert(self, solution: List[List[int]], nodes: List[int]):
        for n in nodes:
            best_cost = float('inf')
            best_pos = None
            best_route = None
            for r_idx, route in enumerate(solution):
                load = sum(self.demands[x] for x in route[1:-1])
                if load + self.demands[n] > self.capacity:
                    continue
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [n] + route[pos:]
                    c = self.route_cost(new_route)
                    if c < best_cost:
                        best_cost = c
                        best_pos = pos
                        best_route = r_idx
            if best_route is None:
                # create new route
                solution.append([self.depot, n, self.depot])
            else:
                route = solution[best_route]
                solution[best_route] = route[:best_pos] + [n] + route[best_pos:]

    def iterate(self, solution: List[List[int]]) -> List[List[int]]:
        sol = copy.deepcopy(solution)
        if random.random() < 0.5:
            removed = random_removal(sol, max(1, len(self.customers)//10))
        else:
            removed = worst_distance_removal(sol, self.dist, max(1, len(self.customers)//10))

        for idx, r in enumerate(sol):
            sol[idx] = two_opt(r)
        swap_between_routes(sol)
        self.greedy_insert(sol, removed)
        return sol

    def solve(self, iterations: int = 1000) -> Tuple[List[List[int]], float]:
        current = self.initial_solution()
        best = copy.deepcopy(current)
        best_cost = self.evaluate(best)

        for _ in range(iterations):
            candidate = self.iterate(current)
            cand_cost = self.evaluate(candidate)
            if cand_cost < best_cost:
                best = copy.deepcopy(candidate)
                best_cost = cand_cost
            current = candidate
        return best, best_cost
