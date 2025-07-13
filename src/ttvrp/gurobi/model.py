import gurobipy as gp
from gurobipy import GRB
import numpy as np


class TTVRPModel:
    def __init__(self, stops, depots, params):
        """
        初始化TTVRP模型
        params: {
            'vehicle_capacity': 车辆容量,
            'num_vehicles': 车辆数量
        }
        """
        self.stops = stops  # 停靠点DataFrame
        self.depots = depots  # 甩柜点DataFrame
        self.params = params
        self.depot_center = [0, 0]  # 配送中心坐标
        self.model = None
        self.I = None  # 所有服务点集合
        self.Ix = None  # 停靠点集合
        self.Iy = None  # 甩柜点集合
        self.K = None  # 车辆集合
        self.d = None  # 距离矩阵
        self.q = None  # 需求量

    def build_model(self):
        self._prepare_data()
        self.model = gp.Model("TTVRP")

        # 决策变量
        x = self.model.addVars(self.I, self.I, self.K, vtype=GRB.BINARY, name="x")
        y = self.model.addVars(self.I, self.K, vtype=GRB.BINARY, name="y")
        z = self.model.addVars(self.K, vtype=GRB.BINARY, name="z")
        u = self.model.addVars(self.I, self.K, vtype=GRB.INTEGER, lb=0, ub=len(self.I), name="u")

        # 目标函数
        self.model.setObjective(
            100000 * z.sum('*') +
            gp.quicksum(self.d[i, j] * x[i, j, k]
                        for i in self.I for j in self.I if i != j
                        for k in self.K),
            GRB.MINIMIZE
        )

        # 添加约束
        self._add_service_constraints(x, y, z)
        self._add_capacity_constraints(y, z)
        self._add_route_constraints(x, y, z)
        self._add_subtour_elimination(x, u)

        return self.model

    def _prepare_data(self):
        """准备模型所需数据"""
        # 构建服务点集合
        self.I = [0] + self.stops.index.tolist() + self.depots.index.tolist()
        self.Ix = self.stops.index.tolist()
        self.Iy = self.depots.index.tolist()
        self.K = list(range(self.params['num_vehicles']))

        # 计算距离矩阵
        self._calculate_distance_matrix()

        # 准备需求量数据
        self.q = {0: 0}  # 配送中心需求为0
        for idx in self.Ix:
            self.q[idx] = self.stops.loc[idx, 'volume']
        for idx in self.Iy:
            self.q[idx] = self.depots.loc[idx, 'volume']

    def _calculate_distance_matrix(self):
        """计算所有点之间的距离矩阵"""
        self.d = {}
        all_points = {0: self.depot_center}  # 配送中心

        # 添加停靠点和甩柜点坐标
        for idx, row in self.stops.iterrows():
            all_points[idx] = [row['x'], row['y']]
        for idx, row in self.depots.iterrows():
            all_points[idx] = [row['x'], row['y']]

        # 计算距离
        for i in self.I:
            for j in self.I:
                if i != j:
                    self.d[i, j] = np.sqrt(
                        (all_points[i][0] - all_points[j][0]) ** 2 +
                        (all_points[i][1] - all_points[j][1]) ** 2
                    )

    def _add_service_constraints(self, x, y, z):
        """添加服务相关约束"""
        # 每个服务点必须被访问一次
        for i in self.Ix + self.Iy:
            self.model.addConstr(
                gp.quicksum(y[i, k] for k in self.K) == 1
            )

        # 甩柜点必须先于停靠点服务
        for k in self.K:
            self.model.addConstr(
                gp.quicksum(x[0, i, k] for i in self.Iy) == z[k]
            )

    def _add_capacity_constraints(self, y, z):
        """添加容量相关约束"""
        for k in self.K:
            self.model.addConstr(
                gp.quicksum(self.q[i] * y[i, k] for i in self.Ix) <=
                self.params['vehicle_capacity']
            )

    def _add_route_constraints(self, x, y, z):
        """添加路径相关约束"""
        # 流量守恒约束
        for k in self.K:
            for i in self.I:
                self.model.addConstr(
                    gp.quicksum(x[j, i, k] for j in self.I if j != i) ==
                    gp.quicksum(x[i, j, k] for j in self.I if j != i)
                )

        # 甩柜点访问约束
        for i in self.Iy:
            self.model.addConstr(
                gp.quicksum(x[j, i, k] for j in self.I if j != i for k in self.K) == 2
            )

    def _add_subtour_elimination(self, x, u):
        """添加子回路消除约束"""
        M = len(self.I)
        for k in self.K:
            for i in self.Ix:
                for j in self.Ix:
                    if i != j:
                        self.model.addConstr(
                            u[j, k] >= u[i, k] + 1 - M * (1 - x[i, j, k])
                        )
