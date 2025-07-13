# ttslp_vrp/src/ttvrp/cdrl/clustering.py

import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import pandas as pd


def compute_total_demand(cluster, df_service):
    """
    计算聚类内所有点的需求量。
    cluster: list/array of (x, y)
    df_service: pd.DataFrame, 包含service_x, service_y, volume等信息
    """
    total_demand = 0
    for (px, py) in cluster:
        # 在 df_service 中找到与 (px, py) 相同坐标的行，提取 volume
        matched = df_service.loc[
            (df_service['service_x'] == px) &
            (df_service['service_y'] == py),
            'volume'
        ]
        if len(matched) > 0:
            total_demand += matched.values[0]
    return total_demand


def select_distant_centers(points, k):
    """
    选择k个相距最远的点作为初始聚类中心。
    points: np.ndarray [n, 2]
    k: int, 需要的中心个数
    """
    n_points = len(points)
    # 随机选择第一个中心
    first_center_index = np.random.choice(n_points)
    centers = [points[first_center_index]]

    for _ in range(1, k):
        # 计算每个点到当前所有中心的最小距离
        dist_to_centers = pairwise_distances(points, np.array(centers)).min(axis=1)
        # 选择距离最远的点作为新的中心
        new_center_index = np.argmax(dist_to_centers)
        centers.append(points[new_center_index])
    return centers


def greedy_clustering(points, *,
                      max_size=2,
                      Q=500,
                      k=None,
                      mode=0,
                      df_service=None):
    """
    贪心聚类函数：
    points: (n, 2) 的 np.ndarray 或 list
    max_size: 每个聚类的最大容量（数量）
    Q: 当 mode=1(停靠点) 时，用于判断聚类内总需求是否超限
    k: 如果为 None，则由 len(points)//max_size 决定，否则固定聚类数
    mode: 用来区分两种情形
         0 -> 甩柜点聚类(对需求量不作限制，只看 max_size)
         1 -> 停靠点聚类(需同时满足 Q 和 max_size)
    df_service: 若需要计算需求量限制，则需要该 DataFrame
    """
    # 若 k 未传入，则按点数//max_size
    if k is None:
        k = max(1, len(points)//max_size)

    # 选择初始聚类中心
    initial_centers = select_distant_centers(points, k)
    cluster_centers = [np.mean([c], axis=0) for c in initial_centers]
    clusters = [[c] for c in initial_centers]

    for p in points:
        # 如果 p 恰好就是已经选作中心的点，则跳过
        if p in initial_centers:
            continue

        # 计算 p 到各中心的距离
        dists = pairwise_distances([p], cluster_centers, metric='euclidean').flatten()
        # 按距离从近到远排序
        sorted_idx = np.argsort(dists)

        added = False
        for idx in sorted_idx:
            if mode == 0:
                # 甩柜点: 仅看 max_size
                if len(clusters[idx]) < max_size:
                    clusters[idx].append(p)
                    cluster_centers[idx] = np.mean(clusters[idx], axis=0)  # 更新中心
                    added = True
                    break
            else:
                # 停靠点: 既要看 max_size，又要看需求量 Q
                if df_service is None:
                    # 如果没有 df_service，就仅仅看 max_size
                    if len(clusters[idx]) < max_size:
                        clusters[idx].append(p)
                        cluster_centers[idx] = np.mean(clusters[idx], axis=0)
                        added = True
                        break
                else:
                    # 检查当前聚类需求量
                    demand_now = compute_total_demand(clusters[idx], df_service)
                    if (demand_now < Q) and (len(clusters[idx]) < max_size):
                        clusters[idx].append(p)
                        cluster_centers[idx] = np.mean(clusters[idx], axis=0)
                        added = True
                        break

        # 如果没能加进任何现有聚类，而且当前聚类数还不足k，就新建一个聚类
        if not added and len(clusters) < k:
            clusters.append([p])
            cluster_centers.append(np.mean([p], axis=0))

    return clusters


def find_optimal_clustering(points, k,
                            *,
                            iterations=100,
                            mode=0,
                            max_size=2,
                            Q=500,
                            df_service=None):
    """
    多次迭代寻找最优聚类, 以最小方差为指标
    points: 待聚类的 (n,2)
    k: 聚类数
    iterations: 迭代次数
    mode: 0 -> 甩柜点, 1 -> 停靠点
    max_size: 每个聚类容量上限
    Q: 停靠点时用于需求量限制
    df_service: 计算需求量时用到
    """
    best_clusters = None
    min_variance = float('inf')

    # 多次迭代
    for _ in range(iterations):
        random.shuffle(points)  # 打乱点顺序
        clusters = greedy_clustering(
            points,
            max_size=max_size,
            Q=Q,
            k=k,
            mode=mode,
            df_service=df_service
        )
        # 计算总方差
        variance = 0.0
        for cluster in clusters:
            if len(cluster) > 0:
                variance += np.var(cluster, axis=0).sum()
        # 记录最佳解
        if variance < min_variance:
            min_variance = variance
            best_clusters = clusters

    return best_clusters


def compute_min_pairwise_distance(group1, group2):
    """
    计算两组点之间的最小距离
    group1, group2: list/array of (x,y)
    """
    if len(group1) == 0 or len(group2) == 0:
        return float('inf')
    group1 = np.array(group1)
    group2 = np.array(group2)
    distances = cdist(group1, group2, metric='euclidean')
    return np.min(distances)
