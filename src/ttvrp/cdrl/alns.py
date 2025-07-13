"""Adaptive Large Neighbourhood Search utilities for TTVRP.

This module refines initial clustered RL-TSP routes by using ALNS.
It defines the solution representation :class:`VRPState` and several
operators tailored for this problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from alns import ALNS, State
from alns.accept import HillClimbing
from alns.select import RandomSelect
from alns.stop import MaxIterations


@dataclass
class VRPState(State):
    """Represents a candidate vehicle routing solution."""

    routes: List[List[int]]
    coords: Dict[int, tuple]
    demand: Dict[int, float]
    stop_ids: Sequence[int]
    depot_ids: Sequence[int]
    dc_id: int
    capacity: float
    unassigned: List[int] = field(default_factory=list)

    def copy(self) -> "VRPState":
        return VRPState(
            routes=[r.copy() for r in self.routes],
            coords=self.coords,
            demand=self.demand,
            stop_ids=self.stop_ids,
            depot_ids=self.depot_ids,
            dc_id=self.dc_id,
            capacity=self.capacity,
            unassigned=self.unassigned.copy(),
        )

    # ------------------------------------------------------------------
    def route_load(self, route: List[int]) -> float:
        return sum(self.demand.get(n, 0) for n in route if n in self.stop_ids)

    def route_distance(self, route: List[int]) -> float:
        dist = 0.0
        for i in range(len(route) - 1):
            x1, y1 = self.coords[route[i]]
            x2, y2 = self.coords[route[i + 1]]
            dist += np.hypot(x2 - x1, y2 - y1)
        return dist

    def feasible(self) -> bool:
        if self.unassigned:
            return False
        for route in self.routes:
            if route[0] != self.dc_id or route[-1] != self.dc_id:
                return False
            if sum(1 for n in route if n in self.depot_ids) < 2:
                return False
            if self.route_load(route) > self.capacity:
                return False
        return True

    # ------------------------------------------------------------------
    def objective(self) -> float:
        penalty = 0.0
        if self.unassigned:
            penalty += 1e6 * len(self.unassigned)
        for route in self.routes:
            if route[0] != self.dc_id or route[-1] != self.dc_id:
                penalty += 1e6
            depot_count = sum(1 for n in route if n in self.depot_ids)
            if depot_count < 2:
                penalty += 1e6
            load = self.route_load(route)
            if load > self.capacity:
                penalty += 1e6 + 100 * (load - self.capacity)
        distance = sum(self.route_distance(r) for r in self.routes)
        return distance + penalty


# ---------------------- destroy operators -----------------------------


def random_removal(
    state: VRPState, rng: np.random.Generator, fraction: float = 0.2
) -> VRPState:
    new = state.copy()
    stops = [n for r in new.routes for n in r if n in new.stop_ids]
    if not stops:
        return new
    num = max(1, int(len(stops) * fraction))
    remove = rng.choice(stops, size=num, replace=False)
    for nid in remove:
        for r in new.routes:
            if nid in r:
                r.remove(nid)
                break
    new.unassigned.extend(remove.tolist())
    return new


def worst_removal(
    state: VRPState, rng: np.random.Generator, fraction: float = 0.2
) -> VRPState:
    new = state.copy()
    candidates = []
    for ridx, route in enumerate(new.routes):
        for idx, node in enumerate(route[1:-1], start=1):
            if node not in new.stop_ids:
                continue
            dist_before = new.route_distance(route)
            tmp = route[:idx] + route[idx + 1 :]
            dist_after = new.route_distance(tmp)
            candidates.append((dist_before - dist_after, ridx, node))
    if not candidates:
        return new
    candidates.sort(reverse=True)
    num = max(1, int(len(candidates) * fraction))
    for _, ridx, node in candidates[:num]:
        new.routes[ridx].remove(node)
        new.unassigned.append(node)
    return new


def reassign_stop(state: VRPState, rng: np.random.Generator) -> VRPState:
    """Move a random stop to another route."""
    if len(state.routes) < 2:
        return state
    new = state.copy()
    src_idx, dst_idx = rng.choice(len(new.routes), size=2, replace=False)
    src_route = new.routes[src_idx]
    stops = [n for n in src_route if n in new.stop_ids]
    if not stops:
        return new
    node = int(rng.choice(stops))
    src_route.remove(node)
    new.unassigned.append(node)
    return greedy_insert(new, rng)


# ---------------------- repair operators ------------------------------


def greedy_insert(state: VRPState, rng: np.random.Generator) -> VRPState:
    new = state.copy()
    while new.unassigned:
        nid = new.unassigned.pop(0)
        best = None
        best_cost = float("inf")
        demand = new.demand.get(nid, 0)
        for ridx, route in enumerate(new.routes):
            if new.route_load(route) + demand > new.capacity:
                continue
            for pos in range(1, len(route)):
                cand = route[:pos] + [nid] + route[pos:]
                cost = new.route_distance(cand)
                if cost < best_cost:
                    best_cost = cost
                    best = (ridx, pos)
        if best is not None:
            r, p = best
            new.routes[r].insert(p, nid)
        else:
            # new route starting/ending at dc
            new.routes.append([new.dc_id, nid, new.dc_id])
    return new


def regret_insert(state: VRPState, rng: np.random.Generator, k: int = 2) -> VRPState:
    new = state.copy()
    while new.unassigned:
        best_choice = None
        best_regret = -float("inf")
        for nid in new.unassigned:
            demand = new.demand.get(nid, 0)
            positions = []
            for ridx, route in enumerate(new.routes):
                if new.route_load(route) + demand > new.capacity:
                    continue
                inserts = []
                for pos in range(1, len(route)):
                    cand = route[:pos] + [nid] + route[pos:]
                    cost = new.route_distance(cand)
                    inserts.append((cost, pos))
                if inserts:
                    inserts.sort()
                    positions.append((ridx, inserts))
            if not positions:
                continue
            # compute regret value
            costs = []
            for ridx, ins in positions:
                costs.append(ins[0][0])
            costs.sort()
            if len(costs) >= k:
                regret = costs[k - 1] - costs[0]
            else:
                regret = costs[-1] - costs[0]
            if regret > best_regret:
                best_regret = regret
                chosen_route, pos = positions[0][0], positions[0][1][0][1]
                best_choice = (nid, chosen_route, pos)
        if best_choice is None:
            # fallback to greedy
            return greedy_insert(new, rng)
        nid, ridx, pos = best_choice
        new.unassigned.remove(nid)
        new.routes[ridx].insert(pos, nid)
    return new


def two_opt(state: VRPState, rng: np.random.Generator) -> VRPState:
    new = state.copy()
    ridx = rng.integers(len(new.routes))
    route = new.routes[ridx]
    if len(route) <= 4:
        return new
    i, j = sorted(rng.choice(range(1, len(route) - 1), size=2, replace=False))
    new.routes[ridx] = route[:i] + list(reversed(route[i:j])) + route[j:]
    return new


# ----------------------------- solver --------------------------------


class ALNSOptimizer:
    """ALNS optimizer to improve RL routes."""

    def __init__(self, capacity: float = 500):
        self.capacity = capacity

    def optimise(
        self,
        df_service: pd.DataFrame,
        initial_routes: List[List[int]],
        iterations: int = 1000,
        seed: int | None = None,
    ) -> VRPState:
        coords = df_service.set_index("service_id")[["service_x", "service_y"]].to_dict(
            "index"
        )
        demand = df_service.set_index("service_id")["volume"].fillna(0).to_dict()
        stop_ids = df_service[df_service["service_type"] == "停靠点"][
            "service_id"
        ].tolist()
        depot_ids = df_service[df_service["service_type"] == "甩柜点"][
            "service_id"
        ].tolist()
        dc_id = df_service[df_service["service_type"] == "配送中心"]["service_id"].iloc[
            0
        ]

        init_state = VRPState(
            initial_routes, coords, demand, stop_ids, depot_ids, dc_id, self.capacity
        )

        rng = np.random.default_rng(seed)
        alns = ALNS(rng=rng)
        alns.add_destroy_operator(random_removal)
        alns.add_destroy_operator(worst_removal)
        alns.add_destroy_operator(reassign_stop)

        alns.add_repair_operator(greedy_insert)
        alns.add_repair_operator(regret_insert)
        alns.add_repair_operator(two_opt)

        select = RandomSelect(rng)
        accept = HillClimbing()
        stop = MaxIterations(iterations)

        result = alns.iterate(init_state, select, accept, stop)
        return result.best_state
