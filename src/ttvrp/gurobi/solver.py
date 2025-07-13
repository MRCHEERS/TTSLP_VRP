# ttslp_vrp/src/ttvrp/gurobi/solver.py

import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gurobipy import Model, GRB, quicksum
from config.config import Config

class TTVRPGurobiSolver:
    """
    读取 data/ttslp/ 下以 'result.xlsx' 结尾的文件(例如 'xxx_result.xlsx')，
    从中解析服务点(停靠点/甩柜点/配送中心)以及其需求量，然后用 Gurobi 求解 TTVRP 问题。
    并将输出包括:
       1) Summary (总距离、车辆数、时间...) 写入 data/ttvrp/gurobi/ttvrp_gurobi_summary.xlsx
       2) 车辆路径图: data/ttvrp/gurobi/png/xxx_gurobi_routes.png
    """

    def __init__(self, config: Config):
        self.config = config
        # 输入目录: data/ttslp/
        self.input_dir = self.config.DATA_DIR['ttslp']

        # 输出目录
        self.output_dir = os.path.join(
            os.path.dirname(self.config.DATA_DIR['raw']),
            'ttvrp2', 'gurobi'
        )
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        # png 子目录
        self.png_dir = os.path.join(self.output_dir, 'png')
        if not os.path.exists(self.png_dir):
            os.makedirs(self.png_dir)

        # summary 文件
        self.summary_path = os.path.join(self.output_dir, "ttvrp_gurobi_summary.xlsx")

        # 一些固定参数(可根据原代码)
        self.Q = 500   # 车辆容量
        self.w1 = 100000
        self.w2 = 1

    def run_gurobi_for_all(self):
        """
        在 input_dir 中查找以 'result.xlsx' 结尾的文件，
        对每个文件构建 TTVRP 模型并用 Gurobi 求解，输出结果、可视化及 summary。
        """
        all_files = [f for f in os.listdir(self.input_dir) if f.endswith('result.xlsx')]
        if not all_files:
            print(f"[TTVRPGurobiSolver] 未找到任何 'result.xlsx' 文件在 {self.input_dir} 下。")
            return

        results = []

        for fname in all_files:
            path = os.path.join(self.input_dir, fname)
            case_id = os.path.splitext(fname)[0]  # 去除 .xlsx
            print(f"[TTVRPGurobiSolver] 开始处理: {case_id}")
            # 下面以 {case_id}_gurobi_routes.png 作为是否已处理的标记
            out_png = os.path.join(self.png_dir, f"{case_id}_gurobi_routes.png")
            if os.path.exists(out_png):
                print(f"[TTVRPGurobiSolver] 已检测到 {out_png}，跳过算例 {case_id}")
                continue
            # 1) 读取Excel, 通常sheet_name='assignment' (请根据实际情况)
            df_assign = pd.read_excel(path, sheet_name='assignment')
            # 这里假设 df_assign 列包含:
            #   service_id, service_type, service_x, service_y, volume(或可选), ...
            #   还可能包含 'customer_id', 'distance' ...
            # 需要解析出 Ix, Iy, c, K, d, q 等

            # ========== 数据解析: 构造点集 I, 停靠点 Ix, 甩柜点 Iy, 配送中心 c=??? ==========
            # 例如:
            #   c = -1 (或其他) 作为配送中心ID
            #   I = set of all service_id
            #   Ix = service_id where service_type='停靠点'
            #   Iy = service_id where service_type='甩柜点'
            #   K = [0,1,2,3,...] 车辆集合(需自定?)

            # 对 df_assign 进行基本处理:
            # 过滤掉 service_id = np.nan (若存在)
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
            all_columns = df_assign.columns.tolist()
            # 填充缺失的列为 NaN
            for col in all_columns:
                if col not in distribution_center:
                    distribution_center[col] = np.nan
            # 将配送中心行追加到 DataFrame
            # 使用 pd.concat 代替 append
            df_assign = pd.concat([df_assign, pd.DataFrame([distribution_center])], ignore_index=True)
            print(f"[TTVRPGurobiSolver] 已添加配送中心行到 {fname}")
            unique_services = df_assign[['service_id','service_type','service_x','service_y','volume']].drop_duplicates()
            unique_services = unique_services[unique_services['service_id'].notna()]
            # 构造 I
            I = unique_services['service_id'].unique().tolist()
            I = [int(x) for x in I]  # 确保 int

            # 找到停靠点 (Ix)
            Ix = unique_services[ (unique_services['service_type']=='停靠点') ]['service_id'].unique().tolist()
            # 找到甩柜点 (Iy)
            Iy = unique_services[ (unique_services['service_type']=='甩柜点') ]['service_id'].unique().tolist()

            # 假设配送中心 c=0, 并在 df_assign 里找 service_id=0 ?
            # 具体看你 TTSLP 的定义
            # 这里示例:
            c = -1
            # 车辆集合，车辆数取甩柜点数除以2+3
            K = [k for k in range(len(Iy)//2 + 3)]  # 假设3辆车, 由你决定

            # 计算距离 d[i,j], 例如用欧几里得或曼哈顿
            # 需要先构造 service_id -> (x,y) 的映射
            service_coord_map = {}
            for rowi, row in unique_services.iterrows():
                sid = int(row['service_id'])
                service_coord_map[sid] = (row['service_x'], row['service_y'])

            def dist(i, j):
                (x1, y1) = service_coord_map[i]
                (x2, y2) = service_coord_map[j]
                return math.hypot(x2 - x1, y2 - y1)  # 欧几里得

            # 构造 q[i], 仅对停靠点?
            # 或 i in Ix+Iy: volume
            # 视需求:
            q = {}
            for rowi, row in unique_services.iterrows():
                sid = int(row['service_id'])
                # row['volume'] 可能 NaN, 对甩柜点/停靠点需?
                # 在原VRP中, 只对 Ix(停靠点) 有需求?
                if row['volume'] is not None and not np.isnan(row['volume']):
                    q[sid] = row['volume']
                else:
                    q[sid] = 0

            # ========== 构建Gurobi模型: 与原代码保持一致 ==========
            model = Model(f"TTVRP_{case_id}")
            model.setParam('OutputFlag', 1)

            # 决策变量
            x = model.addVars(I, I, K, vtype=GRB.BINARY, name="x")
            y = model.addVars(I, K, vtype=GRB.BINARY, name="y")
            z = model.addVars(K, vtype=GRB.BINARY, name="z")
            # 这里 u(i,k) 也需要
            u = model.addVars(I, K, vtype=GRB.INTEGER, lb=0, ub=len(I), name="u")

            # 目标函数
            model.setObjective(
                self.w1 * z.sum('*')
                + self.w2 * quicksum(dist(i,j) * x[i,j,k] for i in I for j in I if i != j for k in K),
                GRB.MINIMIZE
            )

            # ========== 按您的原约束一一添加 ==========
            # 1) 服务点需求满足(每个 i in Ix+Iy?)
            for i_ in Ix + Iy:
                model.addConstr(quicksum(y[i_, k] for k in K) == 1, name=f"service_requirement_{i_}")

            # 2) 车辆容量
            for k_ in K:
                model.addConstr(quicksum(q[i_]*y[i_, k_] for i_ in Ix) <= self.Q, name=f"capacity_{k_}")

            # 3) 甩柜点率先服务约束
            for k_ in K:
                model.addConstr(quicksum(x[c,i_,k_] for i_ in Iy) == z[k_], name=f"single_initial_trip_for_vehicle_{k_}")
            # 4) 甩柜点结束服务约束
            for k_ in K:
                model.addConstr(quicksum(x[i_,c,k_] for i_ in Iy) == z[k_], name=f"single_final_trip_back_for_vehicle_{k_}")
            # 5) 甩柜点服务对称约束
            for k_ in K:
                for i_ in Iy:
                    model.addConstr(x[c, i_, k_] - x[i_, c, k_] == 0, name=f"initial_trip_symmetry_{k_}_{i_}")

            # 6) 甩柜点进入2次，离开2次
            for i_ in Iy:
                model.addConstr(quicksum(x[j, i_, k_] for j in I if j!=i_ for k_ in K) == 2*z[k_], name=f"enter_dp_point_{i_}_twice")
                model.addConstr(quicksum(x[i_, j, k_] for j in I if j!=i_ for k_ in K) == 2*z[k_], name=f"exit_dp_point_{i_}_twice")

            # 7) 禁止甩柜点和停靠点之间相互访问
            for k_ in K:
                for i_ in Iy:
                    for j_ in Ix:
                        model.addConstr(x[i_, j_, k_] + x[j_, i_, k_] <= 1, name=f"no_mutual_visit_iy_ix_{i_}_{j_}_{k_}")

            # 8) 甩柜点与停靠点连接约束
            for k_ in K:
                model.addConstr(quicksum(x[i_, j_, k_] for i_ in Iy for j_ in Ix) == z[k_], name=f"depot_to_stop_conn_{k_}")
                model.addConstr(quicksum(x[j_, i_, k_] for i_ in Iy for j_ in Ix) == z[k_], name=f"stop_to_depot_conn_{k_}")

            # 9) 服务点数量约束
            for k_ in K:
                model.addConstr(quicksum(y[i_, k_] for i_ in Iy) <= 2*z[k_], name=f"depot_service_count_{k_}")
                model.addConstr(quicksum(y[i_, k_] for i_ in Ix) >= z[k_], name=f"stop_service_minimum_{k_}")

            # 10) 服务连续性
            for k_ in K:
                for i_ in I:
                    model.addConstr(quicksum(x[j_, i_, k_] for j_ in I if j_!=i_)
                                    == quicksum(x[i_, j_, k_] for j_ in I if j_!=i_),
                                    name=f"continuity_{i_}_{k_}")

            # 11) 子回路约束 (对停靠点?)
            M = 1000
            for k_ in K:
                for i_ in Ix:
                    for j_ in Ix:
                        if i_ != j_:
                            model.addConstr(u[j_, k_] >= u[i_, k_] + 1 - M*(1 - x[i_, j_, k_]),
                                            name=f"subtour_elimination_{i_}_{j_}_{k_}")

            # 12) 基于x定义y的约束
            for k_ in K:
                for i_ in Ix + Iy:
                    model.addConstr(quicksum(x[i_, j_, k_] for j_ in I if j_!=i_)
                                    + quicksum(x[j_, i_, k_] for j_ in I if j_!=i_)
                                    >= y[i_, k_],
                                    name=f"define_service_{i_}_{k_}")
            for k_ in K:
                for i_ in I:
                    for j_ in I:
                        if i_ != j_:
                            model.addConstr(x[i_, j_, k_] <= y[i_, k_])
                            model.addConstr(x[j_, i_, k_] <= y[i_, k_])

            # 13) 基于x定义z的约束
            for k_ in K:
                for i_ in I:
                    for j_ in I:
                        if i_ != j_:
                            model.addConstr(x[i_, j_, k_] <= z[k_])
                model.addConstr(z[k_] <= quicksum(x[i_, j_, k_] for i_ in I for j_ in I if i_!=j_),
                                name=f"vehicle_usage_activation_{k_}")

            # ========== 求解 ==========
            #设置时间限制
            model.setParam('TimeLimit', 600)
            t1 = time.time()
            model.optimize()
            used_time = time.time() - t1

            model.optimize()

            if model.SolCount == 0:
                # 没有可行解
                print(f"[TTVRPGurobiSolver] {case_id} 无可行解. (SolCount=0)")
                results.append({
                    'case_id': case_id,
                    'total_distance': -1,
                    'vehicle_num': -1,
                    'time': 1000,
                    'detail_path': '无可行解'
                })
                continue
            else:
                # 至少有一条可行解
                if model.status == GRB.OPTIMAL:
                    print(f"[TTVRPGurobiSolver] {case_id} 求解完成, objVal={model.objVal:.2f}")
                elif model.status == GRB.TIME_LIMIT:
                    print(f"[TTVRPGurobiSolver] {case_id} 达到时间限制, 但已有可行解, 当前objVal={model.objVal:.2f}")
                else:
                    print(f"[TTVRPGurobiSolver] {case_id} status={model.status}, 但已有解, objVal={model.objVal:.2f}")
            # ========== 解析解并构建车辆路径 ==========
            # 解析解 -> 路径
            paths = self._parse_solution(I, Ix, Iy, c, K, x, y, z, df_assign)
            # 计算总距离(如按欧几里得或曼哈顿)
            total_dist, used_vehicles = self._calc_total_distance(paths, dist)
            self._plot_final_routes(case_id, paths, df_assign)
            result=[{
                'case_id': case_id,
                'total_distance': total_dist,
                'vehicle_num': used_vehicles,
                'time': used_time,
                'detail_path': 'See console or figure'
            }]
            # 记录
            results.append({
                'case_id': case_id,
                'total_distance': total_dist,
                'vehicle_num': used_vehicles,
                'time': used_time,
                'detail_path': 'See console or figure'
            })
            print(f"[TTVRPGurobiSolver] {case_id} 求解完成. dist={total_dist}, vehicles={used_vehicles}, time={used_time:.2f}s.")
            # 写 Summary
            self._save_summary(result)
        self._save_summary(results)
        print("[TTVRPGurobiSolver] 全部Gurobi求解完成.")

    def _parse_solution(self, I, Ix, Iy, c, K, x, y, z, df_assign):
        """
        从 x[i,j,k], y[i,k], z[k] 解析出车辆路径:
        根据您给出的“甩柜点先行2次, 再停靠点, 回配送中心”的原逻辑构建
        返回: paths = {k: [节点ID...], ...}
        """
        paths = {}
        for k_ in K:
            if z[k_].X < 0.5:
                continue
            # 初始化
            paths[k_] = [c]  # 起点是配送中心 c
            current_point = c
            dropoff_points_added = 0

            # A) 先尝试添加2个甩柜点
            while True:
                if dropoff_points_added < 2:
                    found_next = False
                    for i_ in Iy:
                        if x[current_point, i_, k_].X > 0.5:
                            paths[k_].append(i_)
                            dropoff_points_added += 1
                            current_point = i_
                            found_next = True
                            break
                    if not found_next:
                        break
                elif dropoff_points_added == 2:
                    # 添加停靠点
                    found_stop = False
                    for i_ in Ix:
                        if x[current_point, i_, k_].X > 0.5:
                            paths[k_].append(i_)
                            current_point = i_
                            found_stop = True
                            break
                    if not found_stop:
                        break
                else:
                    break

            # B) 继续找下一个点, 直到回到 c
            while True:
                next_point = None
                for j_ in I:
                    if x[current_point, j_, k_].X > 0.5:
                        # 如果 j_是甩柜点, 且在 paths里出现2次, 则不再加入
                        if j_ in Iy and paths[k_].count(j_) == 2:
                            continue
                        # 如果 j_是停靠点, 且已出现在 path, 就不再加
                        if j_ in Ix and j_ in paths[k_]:
                            continue
                        next_point = j_
                        paths[k_].append(j_)
                        current_point = j_
                        break
                if not next_point or current_point == c:
                    break
            # 打印
            print(f"[TTVRPGurobiSolver] Vehicle={k_}, path={paths[k_]}")
        return paths

    def _calc_total_distance(self, paths, dist_func):
        """
        根据 parse_solution 构建的 paths[k], 用 dist_func(i,j) 计算所有车辆总距离
        并统计使用的车辆数
        """
        total_dist = 0
        used_vehicles = 0
        for k_, route in paths.items():
            if len(route) <= 1:
                continue
            used_vehicles += 1
            dist_sum = 0
            for i_ in range(len(route) - 1):
                dist_sum += dist_func(route[i_], route[i_ + 1])
            total_dist += dist_sum
        return total_dist, used_vehicles

    def _plot_final_routes(self, case_id, paths, df_assign):
        """
        绘制多车路径, 并保存 png
        """
        plt.figure(figsize=(8, 8))
        color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple']
        # 画出所有点
        # df_assign 里包含 service_x, service_y, service_id, service_type
        plt.scatter(df_assign['service_x'], df_assign['service_y'], c='gray', alpha=0.5)

        for k_idx, k_ in enumerate(paths):
            route = paths[k_]
            col = color_list[k_idx % len(color_list)]
            for i_ in range(len(route) - 1):
                sid1 = route[i_]
                sid2 = route[i_ + 1]
                # 取坐标
                row1 = df_assign[df_assign['service_id'] == sid1].iloc[0]
                row2 = df_assign[df_assign['service_id'] == sid2].iloc[0]
                x1, y1 = row1['service_x'], row1['service_y']
                x2, y2 = row2['service_x'], row2['service_y']
                plt.plot([x1, x2], [y1, y2], marker='o', c=col, label=f"Veh{k_}" if i_ == 0 else "")

        plt.title(f"Gurobi TTVRP - {case_id}")
        plt.legend()
        out_png = os.path.join(self.png_dir, f"{case_id}_gurobi_routes.png")
        plt.savefig(out_png)
        plt.close()

    def _save_summary(self, results):
        df_new = pd.DataFrame(results)
        if os.path.exists(self.summary_path):
            df_old = pd.read_excel(self.summary_path)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all.to_excel(self.summary_path, index=False)
        print(f"[TTVRPGurobiSolver] 写入TTVRP-Gurobi结果: {self.summary_path}")

def main():
    config = Config()
    solver = TTVRPGurobiSolver(config)
    solver.run_gurobi_for_all()

if __name__=="__main__":
    main()
