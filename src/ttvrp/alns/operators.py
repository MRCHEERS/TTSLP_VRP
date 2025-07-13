import random
import copy
from typing import List

# Basic solution representation: list of routes, each route is list of node ids.
# index 0 and last of each route is depot id


def random_removal(solution: List[List[int]], num_remove: int) -> List[int]:
    """Randomly remove num_remove customers from the solution."""
    removed = []
    candidate_positions = []
    for r_idx, r in enumerate(solution):
        for pos in range(1, len(r) - 1):
            candidate_positions.append((r_idx, pos))
    random.shuffle(candidate_positions)
    for r_idx, pos in candidate_positions[:num_remove]:
        removed.append(solution[r_idx].pop(pos))
    return removed


def worst_distance_removal(solution: List[List[int]], dist_matrix, num_remove: int) -> List[int]:
    """Remove nodes with largest contribution to distance."""
    contributions = []
    for r_idx, route in enumerate(solution):
        for i in range(1, len(route) - 1):
            prev_n = route[i - 1]
            n = route[i]
            next_n = route[i + 1]
            cost = dist_matrix[prev_n][n] + dist_matrix[n][next_n] - dist_matrix[prev_n][next_n]
            contributions.append((cost, r_idx, i))
    contributions.sort(reverse=True)
    removed = []
    for _, r_idx, pos in contributions[:num_remove]:
        removed.append(solution[r_idx].pop(pos))
    return removed


def two_opt(route: List[int]) -> List[int]:
    if len(route) <= 4:
        return route
    i = random.randint(1, len(route) - 3)
    j = random.randint(i + 1, len(route) - 2)
    return route[:i] + list(reversed(route[i:j])) + route[j:]


def swap_between_routes(solution: List[List[int]]):
    routes = [r for r in solution if len(r) > 2]
    if len(routes) < 2:
        return
    r1, r2 = random.sample(routes, 2)
    i = random.randint(1, len(r1) - 2)
    j = random.randint(1, len(r2) - 2)
    r1[i], r2[j] = r2[j], r1[i]


def shaw_removal(solution: List[List[int]], dist_matrix, num_remove: int) -> List[int]:
    """Remove related customers based on distance (Shaw removal)."""
    customers = [n for r in solution for n in r[1:-1]]
    if not customers:
        return []
    removed = []
    seed = random.choice(customers)
    removed.append(seed)
    while len(removed) < num_remove:
        remaining = list(set(customers) - set(removed))
        if not remaining:
            break
        next_node = min(
            remaining,
            key=lambda n: min(dist_matrix[n][r] for r in removed)
        )
        removed.append(next_node)
    for n in removed:
        for route in solution:
            if n in route:
                route.remove(n)
                break
    return removed


def regret_insert(solution: List[List[int]], nodes: List[int], dist_matrix, demands, capacity, depot=0):
    """Insert nodes using a regret-2 heuristic."""
    while nodes:
        best_regret = -1
        best_choice = None
        for n in nodes:
            best_costs = []
            best_positions = []
            for r_idx, route in enumerate(solution):
                load = sum(demands[x] for x in route[1:-1])
                if load + demands[n] > capacity:
                    continue
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [n] + route[pos:]
                    c = 0
                    for i in range(len(new_route)-1):
                        c += dist_matrix[new_route[i]][new_route[i+1]]
                    best_costs.append(c)
                    best_positions.append((r_idx, pos, c))
            if not best_positions:
                continue
            best_positions.sort(key=lambda x: x[2])
            if len(best_positions) == 1:
                regret = 1e9
            else:
                regret = best_positions[1][2] - best_positions[0][2]
            if regret > best_regret:
                best_regret = regret
                best_choice = (n, best_positions[0])
        if best_choice is None:
            # create new route for random node
            n = nodes.pop(0)
            solution.append([depot, n, depot])
            continue
        n, (r_idx, pos, _) = best_choice
        route = solution[r_idx]
        solution[r_idx] = route[:pos] + [n] + route[pos:]
        nodes.remove(n)
