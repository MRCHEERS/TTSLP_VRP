# ttslp_vrp/src/ttslp/model.py

import numpy as np
import pandas as pd
import math
from scipy.spatial.distance import cdist


class TspMap:
    """
    地图类，用于存储节点坐标和距离矩阵
    """
    def __init__(self, node_positions):
        self.coords = node_positions  # list of tuples (x, y)
        self.node_num = len(node_positions)
        self.dist_table = np.zeros((self.node_num, self.node_num))
        # self._compute_distance_table()

    def _compute_distance_table(self):
        for i in range(self.node_num):
            for j in range(self.node_num):
                dx = self.coords[i][0] - self.coords[j][0]
                dy = self.coords[i][1] - self.coords[j][1]
                self.dist_table[i][j] = math.sqrt(dx*dx + dy*dy)

def generate_candidate_service_points(df_roads,map_size, interval=10):
    """
    根据道路信息，每10米在道路上生成一个候选服务点
    假设道路是水平或垂直的
    """
    candidate_service_points = []

    for _, road in df_roads.iterrows():
        road_name = road['Road_Name']
        sub_name = road['Sub_Name']
        x = road['x']
        y = road['y']

        # 解析道路名称，假设格式如 "Horizontal Road500" 或 "Vertical Road1000"
        if "Horizontal" in road_name:
            # 水平道路，固定y，变化x
            # 假设道路是从 (0, y) 到 (map_size, y)
            # 需要根据地图尺寸动态获取终点
            # 这里假设 map_size 是 3000，实际应根据实际地图尺寸获取
            map_size = map_size  # 您可以将其作为参数传入或从配置中读取
            y_fixed = y
            start_x = 0
            end_x = map_size
            num_points = (end_x - start_x) // interval
            for i in range(num_points + 1):
                new_x = start_x + i * interval
                candidate_service_points.append((new_x, y_fixed))
        elif "Vertical" in road_name:
            # 垂直道路，固定x，变化y
            map_size = map_size  # 同上
            x_fixed = x
            start_y = 0
            end_y = map_size
            num_points = (end_y - start_y) // interval
            for i in range(num_points + 1):
                new_y = start_y + i * interval
                candidate_service_points.append((x_fixed, new_y))
        else:
            # 处理其他类型的道路，如果有的话
            continue

    # 去重
    candidate_service_points = list(set(candidate_service_points))
    # 排序（可选）
    candidate_service_points.sort()
    return candidate_service_points


def compute_distance_matrix(candidate_service_points, df_customer):
    """
    使用 SciPy 的 cdist 来计算距离矩阵:
    d[i, j] = 第 i 个候选服务点 与 第 j 个顾客点 的欧几里得距离
    """
    # 将 candidate_service_points 转为 shape=(N,2) 的 np.array
    points_candidate = np.array(candidate_service_points)  # Nx2
    # 将 df_customer 的坐标提取成 shape=(M,2) 的 np.array
    points_customer = df_customer[['x', 'y']].values  # Mx2

    # 直接用 cdist 计算两组点的欧几里得距离(NxM)
    d = cdist(points_candidate, points_customer, metric='euclidean')
    return d
