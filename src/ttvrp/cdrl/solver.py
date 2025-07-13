# ttslp_vrp/src/ttvrp/cdrl/solver.py
# 注意：可放在与上面 CDRLClusteringSolver 同一个文件，也可拆分多个文件。
# 这里只是演示
# ttslp_vrp/src/ttvrp/cdrl/solver.py
# ttslp_vrp/src/ttvrp/cdrl/solver.py

# ttslp_vrp/src/ttvrp/cdrl/solver.py

import os
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.ttvrp.cdrl.rl import Qlearning_tsp
# 引入修订后的 clustering.py
from src.ttvrp.cdrl.clustering import (
    find_optimal_clustering,
    compute_min_pairwise_distance
)
from scipy.optimize import linear_sum_assignment
from config.config import Config

class CDRLClusteringSolver:
    """
    第一阶段：对停靠点 (Ix) 和甩柜点 (Iy) 分别做聚类，再做最优匹配，
    最后将匹配后的 cluster_id 写回 df_service，并存到 *_clustered.xlsx 文件。
    还会在同目录下的 cdrl_clustering_summary.xlsx 中追加记录聚类信息。
    """

    def __init__(self, config: Config):
        self.config = config
        # 输出目录：data/ttvrp/clustered/
        self.clustered_dir = os.path.join(
            os.path.dirname(self.config.DATA_DIR['raw']),
            'ttvrp2', 'clustered'
        )
        if not os.path.exists(self.clustered_dir):
            os.makedirs(self.clustered_dir)
            print(f"[CDRLClusteringSolver] 创建目录: {self.clustered_dir}")

        # 聚类结果信息文件
        self.summary_xlsx_path = os.path.join(
            self.clustered_dir, 'cdrl_clustering_summary.xlsx'
        )

    def run_clustering(self, input_dir=None):
        """
        从 input_dir 下读取每个 .xlsx 文件（假设包含 df_service 信息），
        然后对甩柜点(Iy)、停靠点(Ix)分别聚类，最后用 linear_sum_assignment 做匹配，
        将 cluster_id 写回 df_service 并存到 data/ttvrp/clustered/xxx_clustered.xlsx。
        最后将聚类统计信息写/追加到 cdrl_clustering_summary.xlsx 文件。
        """

        if input_dir is None:
            # 例如，我们假设 TTSLP 求解后的文件都在 data/ttslp/
            input_dir = self.config.DATA_DIR['ttslp']

        all_files = [f for f in os.listdir(input_dir) if f.endswith('result.xlsx')]
        if not all_files:
            print("[CDRLClusteringSolver] 未找到任何输入文件。")
            return

        # 用于收集所有算例的聚类统计信息
        summary_records = []

        for fname in all_files:
            # 先获取基础文件名 (去掉扩展名)
            base_name = os.path.splitext(fname)[0]
            # 构造输出文件名
            output_name = f"{base_name}_clustered.xlsx"
            output_path = os.path.join(self.clustered_dir, output_name)

            # 如果这个算例的聚类结果已存在，跳过
            if os.path.exists(output_path):
                print(f"[CDRLClusteringSolver] 检测到已存在 {output_name}, 跳过: {fname}")
                continue

            file_path = os.path.join(input_dir, fname)
            print(f"[CDRLClusteringSolver] 开始处理: {fname}")

            # 1) 读取每行都为顾客的 DataFrame
            # 其中: 'service_id','service_type','service_x','service_y','volume'
            df_customers = pd.read_excel(file_path)

            # 2) 将顾客行合并成“服务点表”:
            #    对 [service_id, service_type, service_x, service_y] groupby，
            #    对 volume 做 sum() 得到该服务点的总需求。
            df_service = df_customers.groupby(
                ['service_id', 'service_type', 'service_x', 'service_y'],
                as_index=False
            )['volume'].mean()

            # 3) 分别取出 停靠点(Ix) 与 甩柜点(Iy)
            #    注意这里 df_points 不再是“顾客表”，而是“服务点表”
            Ix = df_service[df_service['service_type'] == '停靠点'][['service_x', 'service_y']].values
            Iy = df_service[df_service['service_type'] == '甩柜点'][['service_x', 'service_y']].values

            n_stop_points = len(Ix)
            n_depot_points = len(Iy)

            if n_stop_points == 0 or n_depot_points == 0:
                print(f"[CDRLClusteringSolver] {fname} 中停靠点或甩柜点数量为0，跳过。")
                continue

            start_time = time.time()

            # 1) 甩柜点聚类 (mode=0)
            k_depot = max(1, math.ceil(n_depot_points/2))
            clusters_depot = find_optimal_clustering(
                points=Iy.tolist(),
                k=k_depot,
                iterations=100,
                mode=0,         # 0=>甩柜点
                max_size=2,
                Q=500,
                df_service=df_service
            )
            depot_clusters_count = len(clusters_depot)

            # 2) 停靠点聚类 (mode=1)
            #   k=和甩柜点聚类一样，或者你也可以自定义
            k_stop = depot_clusters_count
            maxsize=math.ceil(n_stop_points/k_stop)+3
            clusters_stop = find_optimal_clustering(
                points=Ix.tolist(),
                k=k_stop,
                iterations=1000,
                mode=1,         # 1=>停靠点
                max_size=maxsize,
                Q=500,
                df_service=df_service
            )
            stop_clusters_count = len(clusters_stop)

            # 3) 计算甩柜点聚类与停靠点聚类质心的最小距离并做最优匹配
            depot_centers = [np.mean(c, axis=0) for c in clusters_depot]
            stop_centers  = [np.mean(c, axis=0) for c in clusters_stop]
            k1, k2 = len(depot_centers), len(stop_centers)
            dist_matrix = np.zeros((k1, k2))
            for i in range(k1):
                for j in range(k2):
                    dist_matrix[i,j] = compute_min_pairwise_distance(clusters_depot[i], clusters_stop[j])

            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            match_count = len(row_ind)  # 一般等于 min(k1,k2)

            # 4) 更新 df_service['cluster_id_mod']
            df_service['cluster_id'] = np.nan
            for cid, (i, j) in enumerate(zip(row_ind, col_ind), start=1):
                # 甩柜点
                for (px, py) in clusters_depot[i]:
                    df_service.loc[
                        (df_service['service_type']=='甩柜点') &
                        (np.isclose(df_service['service_x'], px)) &
                        (np.isclose(df_service['service_y'], py)),
                        'cluster_id'
                    ] = cid
                # 停靠点
                for (px, py) in clusters_stop[j]:
                    df_service.loc[
                        (df_service['service_type']=='停靠点') &
                        (np.isclose(df_service['service_x'], px)) &
                        (np.isclose(df_service['service_y'], py)),
                        'cluster_id'
                    ] = cid
            #df_service加一行配送中心，service_id=-1,service_x=0,service_y=0,service_type=配送中心,其他字段为空
            # 5) 添加配送中心行
            distribution_center = {
                'service_id': -1,
                'service_type': '配送中心',
                'service_x': 0,
                'service_y': 0,
                # 其他字段根据实际情况添加，如果存在其他字段，可以设置为 NaN 或空字符串
                # 例如，如果存在 'volume' 字段:
                'volume': np.nan
            }
            # 确保所有必要的列都在字典中
            # 获取所有列名
            all_columns = df_service.columns.tolist()
            # 填充缺失的列为 NaN
            for col in all_columns:
                if col not in distribution_center:
                    distribution_center[col] = np.nan

            # 将配送中心行追加到 DataFrame
            # 使用 pd.concat 代替 append
            df_service = pd.concat([df_service, pd.DataFrame([distribution_center])], ignore_index=True)
            print(f"[CDRLClusteringSolver] 已添加配送中心行到 {fname}")
            end_time = time.time()
            used_time = end_time - start_time
            print(f"[CDRLClusteringSolver] 聚类 & 匹配完成: {fname}, 耗时={used_time:.2f}s")

            # 保存 {original_file}_clustered.xlsx
            base_name = os.path.splitext(fname)[0]
            output_name = f"{base_name}_clustered.xlsx"
            output_path = os.path.join(self.clustered_dir, output_name)
            df_service.to_excel(output_path, index=False, sheet_name='clustered')
            print(f"[CDRLClusteringSolver] 写出 {output_name} -> {output_path}")

            # 记录本算例的聚类统计信息
            summary_records.append({
                'filename': fname,
                'depot_points': n_depot_points,
                'k_depot': k_depot,
                'depot_clusters_count': depot_clusters_count,
                'stop_points': n_stop_points,
                'k_stop': k_stop,
                'stop_clusters_count': stop_clusters_count,
                'match_count': match_count,
                'used_time': used_time
            })

        # 将 summary_records 写入/追加到 cdrl_clustering_summary.xlsx
        self.save_cluster_summary(summary_records)

    def run_clustering_single(self, file_path):
        """只对指定的 TTSLP 结果文件进行聚类"""
        if not os.path.exists(file_path):
            print(f"[CDRLClusteringSolver] 未找到文件: {file_path}")
            return

        fname = os.path.basename(file_path)
        summary_records = []

        base_name = os.path.splitext(fname)[0]
        output_name = f"{base_name}_clustered.xlsx"
        output_path = os.path.join(self.clustered_dir, output_name)

        if os.path.exists(output_path):
            print(f"[CDRLClusteringSolver] 检测到已存在 {output_name}, 跳过: {fname}")
            return

        df_customers = pd.read_excel(file_path)
        df_service = df_customers.groupby(
            ['service_id', 'service_type', 'service_x', 'service_y'],
            as_index=False
        )['volume'].mean()

        Ix = df_service[df_service['service_type'] == '停靠点'][['service_x', 'service_y']].values
        Iy = df_service[df_service['service_type'] == '甩柜点'][['service_x', 'service_y']].values

        n_stop_points = len(Ix)
        n_depot_points = len(Iy)

        if n_stop_points == 0 or n_depot_points == 0:
            print(f"[CDRLClusteringSolver] {fname} 中停靠点或甩柜点数量为0，跳过。")
            return

        start_time = time.time()

        k_depot = max(1, math.ceil(n_depot_points/2))
        clusters_depot = find_optimal_clustering(
            points=Iy.tolist(),
            k=k_depot,
            iterations=100,
            mode=0,
            max_size=2,
            Q=500,
            df_service=df_service
        )
        depot_clusters_count = len(clusters_depot)

        k_stop = depot_clusters_count
        maxsize = math.ceil(n_stop_points/k_stop) + 3
        clusters_stop = find_optimal_clustering(
            points=Ix.tolist(),
            k=k_stop,
            iterations=1000,
            mode=1,
            max_size=maxsize,
            Q=500,
            df_service=df_service
        )
        stop_clusters_count = len(clusters_stop)

        depot_centers = [np.mean(c, axis=0) for c in clusters_depot]
        stop_centers = [np.mean(c, axis=0) for c in clusters_stop]
        k1, k2 = len(depot_centers), len(stop_centers)
        dist_matrix = np.zeros((k1, k2))
        for i in range(k1):
            for j in range(k2):
                dist_matrix[i, j] = compute_min_pairwise_distance(clusters_depot[i], clusters_stop[j])

        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        match_count = len(row_ind)

        df_service['cluster_id'] = np.nan
        for cid, (i, j) in enumerate(zip(row_ind, col_ind), start=1):
            for (px, py) in clusters_depot[i]:
                df_service.loc[
                    (df_service['service_type'] == '甩柜点') &
                    (np.isclose(df_service['service_x'], px)) &
                    (np.isclose(df_service['service_y'], py)),
                    'cluster_id'
                ] = cid
            for (px, py) in clusters_stop[j]:
                df_service.loc[
                    (df_service['service_type'] == '停靠点') &
                    (np.isclose(df_service['service_x'], px)) &
                    (np.isclose(df_service['service_y'], py)),
                    'cluster_id'
                ] = cid

        distribution_center = {
            'service_id': -1,
            'service_type': '配送中心',
            'service_x': 0,
            'service_y': 0,
            'volume': np.nan
        }
        for col in df_service.columns.tolist():
            if col not in distribution_center:
                distribution_center[col] = np.nan

        df_service = pd.concat([df_service, pd.DataFrame([distribution_center])], ignore_index=True)

        used_time = time.time() - start_time
        df_service.to_excel(output_path, index=False, sheet_name='clustered')
        print(f"[CDRLClusteringSolver] 写出 {output_name} -> {output_path}")

        summary_records.append({
            'filename': fname,
            'depot_points': n_depot_points,
            'k_depot': k_depot,
            'depot_clusters_count': depot_clusters_count,
            'stop_points': n_stop_points,
            'k_stop': k_stop,
            'stop_clusters_count': stop_clusters_count,
            'match_count': match_count,
            'used_time': used_time
        })

        self.save_cluster_summary(summary_records)

    def save_cluster_summary(self, records):
        """
        将每个文件的聚类统计信息追加到 cdrl_clustering_summary.xlsx
        """
        if not records:
            return
        df_new = pd.DataFrame(records)

        # 如果存在，则与旧表拼起来
        if os.path.exists(self.summary_xlsx_path):
            df_old = pd.read_excel(self.summary_xlsx_path)
            df_result = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_result = df_new

        df_result.to_excel(self.summary_xlsx_path, index=False)
        print(f"[CDRLClusteringSolver] 聚类统计已写入 {self.summary_xlsx_path}")

    def main_cluster(self):
        """
        如果想从命令行直接运行: python -m src.ttvrp.cdrl.solver
        然后自动调用 main_cluster() 逻辑
        """
        self.run_clustering()

class CDRLRLSolver:
    """
    第二阶段：TTVRP的RL求解器。
    逻辑：对 cluster_id 的甩柜点先用贪心连接 => path_first,
          最后一个甩柜点 + cluster_id的所有停靠点 => 调用QlearningTSP做TSP => route_tsp
          最终将 route_tsp 拼接到 path_first, 再闭环回到配送中心。
    """

    def __init__(self, config: Config):
        self.config = config
        # 默认从 data/ttvrp/clustered/ 读取
        self.clustered_dir = os.path.join(
            os.path.dirname(self.config.DATA_DIR['raw']),
            'ttvrp2', 'clustered'
        )
        # 输出结果(汇总表, 图片)等可放 data/ttvrp/rl/
        self.rl_output_dir = os.path.join(
            os.path.dirname(self.config.DATA_DIR['raw']),
            'ttvrp2', 'rl'
        )
        if not os.path.exists(self.rl_output_dir):
            os.makedirs(self.rl_output_dir)

        self.summary_xlsx_path = os.path.join(self.rl_output_dir, 'cdrl_rl_summary.xlsx')
        # 若存在,后续可append; 若不存在,后续第一次写入

    def run_rl_for_case(self, filename):
        """只求解指定的 *_clustered.xlsx 文件"""

        if not filename.endswith('_clustered.xlsx'):
            print(f"[CDRLRLSolver] 文件名需以 _clustered.xlsx 结尾: {filename}")
            return

        path = os.path.join(self.clustered_dir, filename)
        if not os.path.exists(path):
            print(f"[CDRLRLSolver] 未找到文件: {path}")
            return

        self._run_single_case(path)

    def _run_single_case(self, path):
        """内部函数: 求解单个聚类文件并记录结果"""
        start_time = time.time()

        fname = os.path.basename(path)
        base_name = os.path.splitext(fname)[0]
        case_id = base_name.replace("_clustered", "")

        df_clustered = pd.read_excel(path, sheet_name='clustered')
        cluster_ids = df_clustered['cluster_id'].unique()

        route_alls = []
        distance_alls = []

        for cid in cluster_ids:
            if np.isnan(cid):
                continue

            cluster_points = df_clustered[df_clustered['cluster_id'] == cid].copy()
            stop_points = cluster_points[cluster_points['service_type'] == '停靠点']
            depot_points = cluster_points[cluster_points['service_type'] == '甩柜点']
            start_point = df_clustered[df_clustered['service_type'] == '配送中心'].squeeze()

            path_first = [(start_point.service_id, start_point.service_x, start_point.service_y)]
            cur_point = start_point
            dpoints = depot_points.copy()

            while len(dpoints) > 0:
                min_distance = float('inf')
                min_depot_point = None
                for idx, row in dpoints.iterrows():
                    dist_ = abs(cur_point['service_x'] - row['service_x']) + abs(cur_point['service_y'] - row['service_y'])
                    if dist_ < min_distance:
                        min_distance = dist_
                        min_depot_point = row
                path_first.append((min_depot_point.service_id, min_depot_point.service_x, min_depot_point.service_y))
                cur_point = min_depot_point
                dpoints = dpoints[dpoints['service_id'] != min_depot_point['service_id']]

            node_positions = [(cur_point['service_x'], cur_point['service_y'])]
            for _, row in stop_points.iterrows():
                node_positions.append((row.service_x, row.service_y))

            rl_node_num = len(node_positions)
            qlearn = Qlearning_tsp(alpha=0.5, gamma=0.01, epsilon=0.5, final_epsilon=0.05,
                                   node_num=rl_node_num, mapsize=400)
            qlearn.tsp_map.read_node_positions(node_positions)
            qlearn.Train_Qtable(iter_num=5000)

            route_tsp = []
            for i in qlearn.good['path']:
                x_ = qlearn.tsp_map.coord_x[i]
                y_ = qlearn.tsp_map.coord_y[i]
                matched_row = cluster_points[(np.isclose(cluster_points['service_x'], x_)) &
                                             (np.isclose(cluster_points['service_y'], y_))]
                if len(matched_row) > 0:
                    sid_ = matched_row.iloc[0]['service_id']
                    route_tsp.append((sid_, x_, y_))
                else:
                    route_tsp.append((-999, x_, y_))

            route_all = path_first + route_tsp[1:]
            route_all = route_all + [route_all[1]] + [route_all[0]]

            route_alls.append(route_all)

            total_distance = 0
            for i_ in range(len(route_all) - 1):
                dx = abs(route_all[i_][1] - route_all[i_ + 1][1])
                dy = abs(route_all[i_][2] - route_all[i_ + 1][2])
                total_distance += (dx + dy)
            distance_alls.append(total_distance)

        self.plot_final_routes(route_alls, case_id)

        used_time = time.time() - start_time
        sum_dist = sum(distance_alls)
        result = {
            'case_id': case_id,
            'total_distance': sum_dist,
            'vehicle_num': len(cluster_ids),
            'time': used_time,
            'detail_path': "See route_alls"
        }

        self.save_summary([result])

    def run_rl_for_all_clustered(self):
        """
        遍历 data/ttvrp/clustered 下所有 *_clustered.xlsx 文件，
        对其进行 RL 路径求解，最后写入 summary，并画出总路径。
        """
        all_files = [f for f in os.listdir(self.clustered_dir) if f.endswith('_clustered.xlsx')]
        if not all_files:
            print("[CDRLRLSolver] 未找到聚类后的文件(*_clustered.xlsx)。请先运行第一阶段聚类。")
            return

        results = []

        for fname in all_files:
            start_time = time.time()

            path = os.path.join(self.clustered_dir, fname)
            base_name = os.path.splitext(fname)[0]  # e.g. C101_clustered
            case_id = base_name.replace("_clustered","")

            # 读取
            df_clustered = pd.read_excel(path, sheet_name='clustered')
            # df_clustered 中应该包含: cluster_id_mod, service_x, service_y, type(?=停靠点/甩柜点/配送中心?)
            cluster_ids = df_clustered['cluster_id'].unique()

            route_alls = []     # 用于记录所有 cluster_id 的最终路径
            distance_alls = []  # 记录每个 cluster_id 的最终距离

            for cid in cluster_ids:
                #如果cluster_id为nan,跳过
                if np.isnan(cid):
                    continue
                # 1) 取属于这个 cluster_id 的点
                cluster_points = df_clustered[df_clustered['cluster_id'] == cid].copy()
                stop_points = cluster_points[cluster_points['service_type'] == '停靠点']
                depot_points = cluster_points[cluster_points['service_type'] == '甩柜点']
                # 假设全局只有一个'配送中心'点
                # 也可 groupby cluster_id if there's multiple
                start_point = df_clustered[df_clustered['service_type'] == '配送中心'].squeeze()

                # 2) 建立一个初始 path_first: 先从配送中心出发,
                #    然后贪心选距离最近的甩柜点依次加进来
                path_first = [(start_point.service_id, start_point.service_x, start_point.service_y)]
                cur_point = start_point
                dpoints = depot_points.copy()

                while len(dpoints) > 0:
                    min_distance = float('inf')
                    min_depot_point = None
                    for idx, row in dpoints.iterrows():
                        dist_ = abs(cur_point['service_x'] - row['service_x']) \
                              + abs(cur_point['service_y'] - row['service_y'])
                        if dist_ < min_distance:
                            min_distance = dist_
                            min_depot_point = row
                    # 加到 path_first
                    path_first.append((min_depot_point.service_id,
                                       min_depot_point.service_x,
                                       min_depot_point.service_y))
                    # 更新 cur_point, 并从 depot_points 移除
                    cur_point = min_depot_point
                    dpoints = dpoints[dpoints['service_id'] != min_depot_point['service_id']]

                print(f"[CDRLRLSolver] cluster_id={cid} 的 path_first(含所有甩柜点) => {path_first}")

                # 3) 最后一个甩柜点(在 cur_point 里) + 所有停靠点 => QlearningTSP
                #    构造 node_positions
                node_positions = [(cur_point['service_x'], cur_point['service_y'])]
                for _, row in stop_points.iterrows():
                    node_positions.append((row.service_x, row.service_y))

                # 4) 调用 RL
                rl_node_num = len(node_positions)
                qlearn = Qlearning_tsp(
                    alpha=0.5, gamma=0.01, epsilon=0.5, final_epsilon=0.05,
                    node_num=rl_node_num, mapsize=400
                )
                qlearn.tsp_map.read_node_positions(node_positions)
                # 可按需求决定是否画图: qlearn.tsp_map.Draw_map()
                qlearn.Train_Qtable(iter_num=5000)

                # 5) 根据 qlearn.good['path'] => route_tsp
                #    还原坐标, 再在 cluster_points 里找到其 service_id
                #    (注: 只有 '最后一个甩柜点' + '停靠点' => map?)
                route_tsp = []
                for i in qlearn.good['path']:
                    x_ = qlearn.tsp_map.coord_x[i]
                    y_ = qlearn.tsp_map.coord_y[i]
                    # 在 cluster_points 里找
                    matched_row = cluster_points[
                        (np.isclose(cluster_points['service_x'], x_)) &
                        (np.isclose(cluster_points['service_y'], y_))
                    ]
                    if len(matched_row) > 0:
                        sid_ = matched_row.iloc[0]['service_id']
                        route_tsp.append((sid_, x_, y_))
                    else:
                        # 可能找不到(如果这个点恰好是最后甩柜点?), fallback
                        # 也可 pass
                        route_tsp.append((-999, x_, y_))

                # 6) 将 route_tsp 与 path_first的前2个点拼接 => route_all
                #    这里根据你原始逻辑：
                #    route_all = path_first + route_tsp[1:]
                #    然后把 route_all 与 path_first[1] + path_first[0] 结尾
                #    (这样是把 path_first[1]当成第一个甩柜点?)
                route_all = path_first + route_tsp[1:]
                # 闭环: + [route_all[1]] + [route_all[0]]
                route_all = route_all + [route_all[1]] + [route_all[0]]

                route_alls.append(route_all)

                # 7) 计算距离(按曼哈顿)
                total_distance = 0
                for i_ in range(len(route_all)-1):
                    dx = abs(route_all[i_][1] - route_all[i_+1][1])
                    dy = abs(route_all[i_][2] - route_all[i_+1][2])
                    total_distance += (dx + dy)
                distance_alls.append(total_distance)

                print(f"[CDRLRLSolver] cluster_id={cid} 最终路径 => {route_all}")
                print(f"[CDRLRLSolver] cluster_id={cid} 的总距离 => {total_distance}")

            # ================== 多个 cluster_id 全部处理完 ==================

            # 画出多条路径
            self.plot_final_routes(route_alls, case_id)

            used_time = time.time() - start_time
            # 这里可以自己定义: total_dist = sum(distance_alls)?
            # vehicles = len(cluster_ids)
            # ...
            sum_dist = sum(distance_alls)
            results.append({
                'case_id': case_id,
                'total_distance': sum_dist,
                'vehicle_num': len(cluster_ids),
                'time': used_time,
                'detail_path': "See route_alls"
            })

            print(f"[CDRLRLSolver] case_id={case_id} done, total_dist={sum_dist}, vehicles={len(cluster_ids)}, time={used_time:.2f}s")

        # 写到 Excel
        self.save_summary(results)

    def plot_final_routes(self, route_alls, case_id):
        """
        绘制多个 cluster_id 的最终路径
        route_alls: [ [ (sid, x, y), ...],  [ (sid, x, y), ... ], ... ]
        """
        plt.figure(figsize=(8,8))
        colors = ['b','g','r','c','m','y','k']*10
        for idx, route_all in enumerate(route_alls):
            xs = [p[1] for p in route_all]
            ys = [p[2] for p in route_all]
            plt.plot(xs, ys, colors[idx], marker='o', label=f"Cluster{idx}")
        plt.title(f"TTVRP RL final - {case_id}")
        plt.legend()
        output_png = os.path.join(self.rl_output_dir, f"{case_id}_final_routes.png")
        plt.savefig(output_png)
        plt.close()

    def save_summary(self, results):
        """
        追加写入 self.summary_xlsx_path
        """
        df_new = pd.DataFrame(results)
        if os.path.exists(self.summary_xlsx_path):
            df_old = pd.read_excel(self.summary_xlsx_path)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all.to_excel(self.summary_xlsx_path, index=False)
        print(f"[CDRLRLSolver] 写入RL求解结果: {self.summary_xlsx_path}")

def main_rl():
    config = Config()
    rl_solver = CDRLRLSolver(config)
    rl_solver.run_rl_for_all_clustered()

# 如果你想在同一个 solver.py 中都能独立运行：
if __name__ == "__main__":
    # 可以分别调用
    config = Config()
    solver = CDRLClusteringSolver(config)
    solver.main_cluster()
    main_rl()         # 再做RL
