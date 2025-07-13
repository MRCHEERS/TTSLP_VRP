# ttslp_vrp/src/ttvrp/cdrl/rl.py
# --------------
# 与您提供的源代码逻辑保持一致的 Q-learning 算法脚本
# 包含 Tsp_Map 与 Qlearning_tsp 两大类，以及末尾测试示例
# --------------

# coding=utf-8
# author:ZP
# create_date:2022/10/31 13:41
# brief:qlearning处理TSP问题。包含Map类和Qlearning类

import numpy as np
import math
import time
import matplotlib.pyplot as plt
import os
import pandas as pd  # 仅末尾测试代码需要
# 如果项目中不需要 df_service 读取，可自行移除 pandas


# -------------------------------
# Tsp_Map 类
# 存储地图尺寸、节点数量、节点坐标、[节点间距离]的信息
# 可随机生成新地图、从txt读取、写入txt、绘制地图、计算节点间距离等
# -------------------------------
class Tsp_Map:
    # 初始化地图尺寸、节点数量、节点坐标，并计算节点间距离
    def __init__(self, map_size=100, node_num=40):
        # np.random.seed(0)  # 随机数种子（若需要固定随机可打开）
        self.map_size = map_size   # 地图尺寸
        self.node_num = node_num   # 节点数量
        self.Init_coords()         # 初始化节点坐标
        self.Calc_distances_table()# 计算各个节点间距离

    # 初始化节点坐标 - 随机产生
    def Init_coords(self):
        self.base_coord = [0, 0]  # 起点坐标
        self.coords = [[0, 0]]    # 每个节点位置坐标(第一个节点为base)
        self.coord_x, self.coord_y = [self.base_coord[0]], [self.base_coord[1]]
        # 随机产生另外 node_num-1 个坐标
        for i in range(self.node_num - 1):
            x, y = np.random.randint(0, self.map_size, size=2)
            while [x, y] in self.coords:
                x, y = np.random.randint(0, self.map_size, size=2)
            self.coord_x.append(x)
            self.coord_y.append(y)
            self.coords.append([x, y])
        print("[地图初始化] {}个节点坐标已生成".format(self.node_num))

    # 计算节点间距离形成距离表 - 根据节点坐标、节点数量
    def Calc_distances_table(self):
        self.distances_table = np.zeros((self.node_num, self.node_num), dtype=float)
        for a in range(self.node_num):
            for b in range(a + 1, self.node_num):
                self.distances_table[a][b] = self.Calc_distance(self.coords[a], self.coords[b])
                self.distances_table[b][a] = self.distances_table[a][b]

    # 计算两点间距离
    def Calc_distance(self, a, b):
        d = math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        return d
    # 从一个传入的 node_positions 列表直接读坐标
    def read_node_positions(self, node_positions):
        self.node_num = len(node_positions)
        for i, point in enumerate(node_positions):
            if i == 0:
                self.base_coord = [point[0], point[1]]
            self.coords[i] = [point[0], point[1]]
            self.coord_x[i] = point[0]
            self.coord_y[i] = point[1]
        self.Calc_distances_table()

    # 绘制地图 根据节点坐标
    def Draw_map(self):
        plt.figure(3)
        plt.title("node_num: {}".format(self.node_num))
        # 画上所有节点
        plt.scatter(self.coord_x, self.coord_y, c="blue", s=50)
        # 地图上标注 BASE点
        xy = self.base_coord  # 得到基地坐标
        xytext = xy[0] - 2, xy[1] - 3
        plt.annotate("BASE", xy=xy, xytext=xytext, weight="bold")
        # Show节点序号
        x, y = [], []
        for i in range(self.node_num):
            x.append(self.coord_x[i])
            y.append(self.coord_y[i])
            xy = (self.coord_x[i], self.coord_y[i])
            xytext = xy[0] + 0.1, xy[1] - 0.05
            plt.annotate(str(i), xy=xy, xytext=xytext, weight="bold")

        try:
            os.mkdir(os.getcwd() + "/" + "png")
            print("[Draw_map] png文件夹创建成功")
        except:
            print("[Draw_map] png文件夹已存在")
        timestr = time.strftime("%Y-%m-%d %H：%M：%S")
        save_path = f"png/map{self.node_num} {timestr}"
        plt.savefig(save_path + '.png', format='png')
        print("[draw map] 地图已保存")
        plt.show()


# -------------------------------
# Qlearning_tsp 类
# 定义Q表、算法训练参数、实现Qlearning流程、选择动作、可视化等
# 与源代码保持一致
# -------------------------------
class Qlearning_tsp:
    # 初始化 dqn 算法参数和地图信息
    def __init__(self, gamma=0.3, alpha=0.3, epsilon=0.9, final_epsilon=0.05, node_num=40, mapsize=100):
        self.tsp_map = Tsp_Map(mapsize, node_num)     # 创建地图类对象
        self.actions = np.arange(0, self.tsp_map.node_num)  # 创建并初始化动作空间
        self.Qtable = np.zeros((self.tsp_map.node_num, self.tsp_map.node_num))  # 创建并初始化Q表

        # 记录训练得到的最优路线和最差路线
        self.good = {'path': [0], 'distance': 0, 'episode': 0}
        self.bad = {'path': [0], 'distance': 0, 'episode': 0}

        # 算法相关参数
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.final_epsilon = final_epsilon

        self.iter_num = 0  # 训练次数记录

    # 训练智能体 s a r s
    def Train_Qtable(self, iter_num=1000):
        gamma = self.gamma
        alpha = self.alpha
        epsilon = self.epsilon
        t1 = time.perf_counter()

        qvalue = self.Qtable.copy()
        plot_dists = []
        plot_iter_nums = []
        self.iter_num = iter_num

        for iter_i in range(iter_num):
            path = []
            s = 0  # 初始状态=0号点(基点)
            path.append(s)
            flag_done = False
            round_dist = 0

            while not flag_done:
                a = self.Choose_action(path, epsilon, qvalue)
                s_next, r, flag_done = self.Transform(path, a)
                round_dist += self.tsp_map.distances_table[s, a]
                path.append(s_next)

                # 更新Qtable
                if flag_done:
                    q_target = r
                    qvalue[s, a] = qvalue[s, a] + alpha * (q_target - qvalue[s, a])
                    break
                else:
                    a1 = self.greedy_policy(path, qvalue)
                    q_target = r + gamma * qvalue[s_next, a1]
                qvalue[s, a] = qvalue[s, a] + alpha * (q_target - qvalue[s, a])
                s = s_next

            # 每一轮探索率epsilon衰减一次
            if epsilon > self.final_epsilon:
                epsilon *= 0.997

            plot_iter_nums.append(iter_i + 1)
            plot_dists.append(round_dist)

            # 记录最好/最坏成绩
            if round_dist <= np.min(plot_dists):
                self.Qtable = qvalue.copy()
                self.good['path'] = path.copy()
                self.good['distance'] = round_dist
                self.good['episode'] = iter_i + 1

            if round_dist >= np.max(plot_dists):
                self.bad['path'] = path.copy()
                self.bad['distance'] = round_dist
                self.bad['episode'] = iter_i + 1

            # 打印进度条
            percent = (iter_i + 1) / iter_num
            bar = '*' * int(percent * 30) + '->'
            delta_t = time.perf_counter() - t1
            pre_total_t = (iter_num * delta_t) / (iter_i + 1)
            left_t = pre_total_t - delta_t
            # print('\r{:6}/{:6}\t训练已完成:{:5.2f}%[{:32}]已用时:{:5.2f}s,预计用时:{:.2f}s,预计剩余:{:.2f}s'
            #       .format((iter_i + 1), iter_num, percent * 100, bar, delta_t,
            #               pre_total_t, left_t), end='')

        print('\n', "qlearning_tsp result".center(40, '='))
        # print('训练中出现的最短路线长度：{},出现在第 {} 次训练中'.format(self.good['distance'], self.good['episode']))
        # print("最短路线:", self.good['path'])
        # print('训练中出现的最长路线长度：{},出现在第 {} 次训练中'.format(self.bad['distance'], self.bad['episode']))
        # print("最长路线:", self.bad['path'])

        # # 绘训练效果图
        # self.Plot_train_process(plot_iter_nums, plot_dists)
        # # 绘制最优路线图
        # self.Plot_path(self.good['path'])

    # 训练曲线可视化
    def Plot_train_process(self, iter_nums, dists):
        plt.figure(1)
        plt.title(f"qlearning node_num:{self.tsp_map.node_num}")
        plt.ylabel("distance")
        plt.xlabel("iteration")
        plt.plot(iter_nums, dists, color='blue')

        try:
            os.mkdir(os.getcwd() + "/" + "png")
            print("[Plot_train_process] png文件夹创建成功")
        except:
            pass
        timestr = time.strftime("%Y-%m-%d %H：%M：%S")
        save_path = f"png/process{self.tsp_map.node_num} {timestr}"
        plt.savefig(save_path + '.png', format='png')
        # plt.show()

    # 绘制最优路线图
    def Plot_path(self, path):
        plt.figure(2)
        plt.title("best route in iter:{}/{}".format(self.good['episode'], self.iter_num) +
                  " Distance:" + "{:.2f}".format(self.good['distance']))
        plt.scatter(self.tsp_map.coord_x, self.tsp_map.coord_y, c="blue", s=50)

        if len(self.good['path']) > 0:
            xy = self.tsp_map.base_coord
            xytext = xy[0] - 4, xy[1] - 5
            plt.annotate("BASE", xy=xy, xytext=xytext, weight="bold")

        if len(path) > 1:
            x, y = [], []
            for i in path:
                x.append(self.tsp_map.coord_x[i])
                y.append(self.tsp_map.coord_y[i])
                xy = (self.tsp_map.coord_x[i], self.tsp_map.coord_y[i])
                xytext = xy[0] + 0.1, xy[1] - 0.05
                plt.annotate(str(i), xy=xy, xytext=xytext, weight="bold")
            plt.plot(x, y, c="red", linewidth=1, linestyle="--")

        try:
            os.mkdir(os.getcwd() + "/" + "png")
            print("[Plot_path] png文件夹创建成功")
        except:
            pass
        timestr = time.strftime("%Y-%m-%d %H：%M：%S")
        save_path = f"png/path{self.tsp_map.node_num} {timestr}"
        plt.savefig(save_path + '.png', format='png')
        plt.show()

    # 环境交互：返回 (s_next, reward, flag_done)
    def Transform(self, path, action):
        # reward = - self.tsp_map.distances_table[int(path[-1]), action]
        # 下面保留源代码中的做法: 距离越大惩罚越大
        reward = -10000 * (self.tsp_map.distances_table[path[-1]][action] / np.max(self.tsp_map.distances_table))

        # 如果已经访问了 node_num 个节点，且 action=0，回到起点 => 完成
        if len(path) == self.tsp_map.node_num and action == 0:
            return action, reward, True
        return action, reward, False

    # 选择动作 (epsilon-greedy)
    def Choose_action(self, path, epsilon, qvalue):
        if len(path) == self.tsp_map.node_num:
            return 0
        q = np.copy(qvalue[path[-1], :])
        if np.random.rand() > epsilon:
            q[path] = -np.inf
            a = np.argmax(q)
        else:
            unvisited = [x for x in self.actions if x not in path]
            a = np.random.choice(unvisited)
        return a

    # 贪心策略
    def greedy_policy(self, path, qvalue):
        if len(path) >= self.tsp_map.node_num:
            return 0
        q = np.copy(qvalue[path[-1], :])
        q[path] = -np.inf
        return np.argmax(q)


# -------------------------------
# 以下是与源代码一起提供的“测试/使用示例”逻辑
# 读取 'clustered37.xlsx' 做聚类ID等处理，然后调用 Qlearning_tsp
# 如果不需要可自行注释
# -------------------------------
if __name__ == '__main__':
    start = time.time()

    df_service = pd.read_excel('clustered37.xlsx')
    # 生成 cluster_id_mod全集，不包括 nan
    cluster_id_mods = df_service['cluster_id_mod'].unique()
    cluster_id_mods = cluster_id_mods[~np.isnan(cluster_id_mods)]

    route_alls = []
    distance_alls = []
    for cluster_id_mod in cluster_id_mods:
        # 选择 cluster_id_mod 对应的停靠点和甩柜点
        cluster_points = df_service[df_service['cluster_id_mod'] == cluster_id_mod]
        stop_points = cluster_points[cluster_points['type'] == '停靠点']
        depot_points = cluster_points[cluster_points['type'] == '甩柜点']
        start_point = df_service[df_service['type'] == '配送中心'].squeeze()

        # 新建一个路径，先从配送中心出发
        path_first = [(start_point.service_id, start_point.service_x, start_point.service_y)]
        # 然后按照贪婪算法，将距离最近的甩柜点依次加入路径
        while len(depot_points) > 0:
            min_distance = float('inf')
            min_depot_point = None
            for index, row in depot_points.iterrows():
                distance_ = abs(start_point['service_x'] - row['service_x']) + abs(
                              start_point['service_y'] - row['service_y'])
                if distance_ < min_distance:
                    min_distance = distance_
                    min_depot_point = row
            path_first.append((min_depot_point.service_id, min_depot_point.service_x, min_depot_point.service_y))
            start_point = min_depot_point
            depot_points = depot_points[depot_points['service_id'] != min_depot_point['service_id']]

        print(f"cluster_id_mod={cluster_id_mod} 的初始路径: {path_first}")

        # 构造 node_positions, 先包含最后一个甩柜点
        node_positions = [(start_point.service_x, start_point.service_y)]
        stop_points = cluster_points[cluster_points['type'] == '停靠点']
        for _, row in stop_points.iterrows():
            node_positions.append((row.service_x, row.service_y))

        numsize = len(node_positions)
        # 创建 Qlearning_tsp 对象
        qlearn = Qlearning_tsp(alpha=0.5, gamma=0.01, epsilon=0.5, final_epsilon=0.05,
                               node_num=numsize, mapsize=400)
        # 用 read_node_positions 替换随机初始化
        qlearn.tsp_map.read_node_positions(node_positions)
        # qlearn.tsp_map.Draw_map()

        # 训练 Q 表
        qlearn.Train_Qtable(iter_num=10000)

        # 根据 best_path 还原坐标 => service_id
        route_tsp = []
        for i in qlearn.good['path']:
            x_ = qlearn.tsp_map.coord_x[i]
            y_ = qlearn.tsp_map.coord_y[i]
            # 根据 x,y 在 cluster_points 中找对应 service_id
            matched = cluster_points[cluster_points['service_x'] == x_]
            # 可能要小心一对多情况，这里假设只有一个匹配
            service_id = matched.squeeze().service_id
            route_tsp.append((service_id, x_, y_))

        # 将 route_tsp 与 path_first 拼接
        route_all = path_first + route_tsp[1:]
        # 再拼接首尾
        route_all = route_all + [route_all[1]] + [route_all[0]]
        route_alls.append(route_all)
        print(f"cluster_id_mod={cluster_id_mod} 最终路径: {route_all}")

        # 计算距离
        total_distance = 0
        for i in range(len(route_all) - 1):
            total_distance += abs(route_all[i][1] - route_all[i+1][1]) + abs(route_all[i][2] - route_all[i+1][2])
        print(f"cluster_id_mod={cluster_id_mod} 的总距离: {total_distance}")
        distance_alls.append(total_distance)

    # 最后汇总可视化
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * 3
    for i, route_all in enumerate(route_alls):
        xs = [p[1] for p in route_all]
        ys = [p[2] for p in route_all]
        plt.plot(xs, ys, colors[i])
    plt.plot(df_service[df_service['type'] == '停靠点']['service_x'],
             df_service[df_service['type'] == '停靠点']['service_y'],
             'o', c='lightblue', label='Stop Points')
    plt.plot(df_service[df_service['type'] == '甩柜点']['service_x'],
             df_service[df_service['type'] == '甩柜点']['service_y'],
             '^', c='lightgreen', label='Depot Points')
    plt.legend(loc='lower left', prop={'size':8})
    plt.show()

    end = time.time()
    print(f"运行时间 = {end - start} 秒")
    print(f"总距离合计 = {sum(distance_alls)}")
