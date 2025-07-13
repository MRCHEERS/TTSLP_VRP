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
