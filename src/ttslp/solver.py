# ttslp_vrp/src/ttslp/solver.py

import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from gurobipy import Model, GRB, quicksum
from src.ttslp.model import TspMap, generate_candidate_service_points, compute_distance_matrix
from config.config import Config
import time

class TTSLPSolver:
    def __init__(self, config: Config):
        self.config = config
        self.ttslp_output_dir = config.DATA_DIR['ttslp']
        self.ensure_directories()
        # summary 文件路径
        self.summary_path = os.path.join(self.ttslp_output_dir, "ttslp_summary.xlsx")

    def ensure_directories(self):
        """确保TTSLP输出目录存在"""
        if not os.path.exists(self.ttslp_output_dir):
            os.makedirs(self.ttslp_output_dir)
            print(f"[TTSLPSolver] 创建目录: {self.ttslp_output_dir}")

    def solve_ttslp_instance(self, instance_path):
        """
        读取单个算例文件，求解TTSLP，并保存结果和可视化图片
        同时记录规模信息到 ttslp_summary.xlsx
        """

        # 1) 读取算例
        with pd.ExcelFile(instance_path) as xls:
            df_customers = pd.read_excel(xls, 'customer')
            df_roads = pd.read_excel(xls, 'road')

        # 2) 生成候选服务点
        map_size_candidates = []
        for _, road in df_roads.iterrows():
            map_size_candidates.append(road['x'])
            map_size_candidates.append(road['y'])
        map_size = int(max(map_size_candidates))  # 假设地图是正方形

        candidate_service_points = generate_candidate_service_points(
            df_roads, interval=20, map_size=map_size
        )
        print(f"[TTSLPSolver] 生成了 {len(candidate_service_points)} 个候选服务点")

        # 3) 创建TspMap对象, 计算距离矩阵
        tsp_map = TspMap(candidate_service_points)
        d = compute_distance_matrix(tsp_map.coords, df_customers)

        n = tsp_map.node_num  # 候选点数量
        m = len(df_customers) # 客户数量
        q = df_customers['volume'].values  # 客户需求量

        # 4) 创建Gurobi模型
        model = Model("TTSLP_Service_Allocation")
        model.setParam('OutputFlag', 1)

        x = model.addVars(n, vtype=GRB.BINARY, name="x")       # 是否为停靠点
        y = model.addVars(n, vtype=GRB.BINARY, name="y")       # 是否为甩柜点
        x_ij = model.addVars(n, m, vtype=GRB.BINARY, name="x_ij")
        y_ij = model.addVars(n, m, vtype=GRB.BINARY, name="y_ij")
        # 在构造 gurobi model 后

        # 引入新的整数变量 z，用来保证 sum(y[i]) 是偶数
        z = model.addVar(vtype=GRB.INTEGER, lb=1, ub=n, name="z")
        obj = quicksum(1000*x[i] for i in range(n)) + quicksum(1000*y[i] for i in range(n)) \
            + quicksum(x_ij[i,j]*d[i,j] for i in range(n) for j in range(m)) \
            + quicksum(y_ij[i,j]*d[i,j] for i in range(n) for j in range(m))
        model.setObjective(obj, GRB.MINIMIZE)

        # 约束
        for j in range(m):
            model.addConstr(quicksum(x_ij[i,j] + y_ij[i,j] for i in range(n))==1, name=f"client_{j}_assign")

        for i in range(n):
            for j in range(m):
                model.addConstr(x_ij[i,j]*d[i,j] <= self.config.TTSLP_PARAMS['R_t'])
                model.addConstr(y_ij[i,j]*d[i,j] <= self.config.TTSLP_PARAMS['R_d'])

        model.addConstr(quicksum(x[i] for i in range(n)) >= 2*quicksum(y[i] for i in range(n)))
        for i in range(n):
            model.addConstr(quicksum(x_ij[i,j]*q[j] for j in range(m)) <= self.config.TTSLP_PARAMS['theta'])
            model.addConstr(quicksum(y_ij[i,j]*q[j] for j in range(m)) <= self.config.TTSLP_PARAMS['rho'])
            model.addConstr(x[i] + y[i] <= 1)
            for j in range(m):
                model.addConstr(x_ij[i,j] <= x[i])
                model.addConstr(y_ij[i,j] <= y[i])



        # 原有约束 ...

        # 最后添加：sum y[i] - 2*z = 0
        model.addConstr(quicksum(y[i] for i in range(n)) - 2 * z == 0, name="y_sum_even")

        # 求解
        start_time = time.time()
        model.setParam('TimeLimit', 600)
        model.optimize()
        end_time = time.time()

        if model.status != GRB.OPTIMAL and model.status != GRB.TIME_LIMIT:
            print(f"[TTSLPSolver] 未找到最优解: {instance_path}")
            return
        selected_stops = [i for i in range(n) if x[i].X > 0.5]
        selected_depots = [i for i in range(n) if y[i].X > 0.5]
        # 5) 构造 assignment 结果
        #    在这里我们同时保留"customer_x, customer_y"以便可视化画虚线
        #    并且先记录customer_volume, 后面再汇总
        assignment_rows = []
        for j in range(m):
            cx = df_customers.loc[j,'x']
            cy = df_customers.loc[j,'y']
            cvol = df_customers.loc[j,'volume']
            for i in range(n):
                if x_ij[i,j].X>0.5:
                    assignment_rows.append({
                        'customer_id': j,
                        'customer_x': cx,
                        'customer_y': cy,
                        'customer_volume': cvol,
                        'service_type': '停靠点',
                        'service_id': i,
                        'service_x': tsp_map.coords[i][0],
                        'service_y': tsp_map.coords[i][1],
                        'distance': d[i,j]
                    })
                if y_ij[i,j].X>0.5:
                    assignment_rows.append({
                        'customer_id': j,
                        'customer_x': cx,
                        'customer_y': cy,
                        'customer_volume': cvol,
                        'service_type': '甩柜点',
                        'service_id': i,
                        'service_x': tsp_map.coords[i][0],
                        'service_y': tsp_map.coords[i][1],
                        'distance': d[i,j]
                    })

        df_assignment = pd.DataFrame(assignment_rows)

        # 6) 对相同 (service_id, service_type) 分组，求总需求量
        #    然后映射回 df_assignment["volume"]
        group_sum = df_assignment.groupby(['service_id','service_type'])['customer_volume'].sum().reset_index()
        group_sum.rename(columns={'customer_volume':'volume'}, inplace=True)  # rename成volume

        # 将其 merge 回 df_assignment，让每行都带上"该服务点的总需求"
        df_assignment = pd.merge(
            df_assignment,
            group_sum,
            on=['service_id','service_type'],
            how='left'
        )

        # 7) 保存
        instance_name = os.path.splitext(os.path.basename(instance_path))[0]
        output_excel_path = os.path.join(self.ttslp_output_dir, f"{instance_name}_result.xlsx")
        with pd.ExcelWriter(output_excel_path) as writer:
            df_assignment.to_excel(writer, sheet_name='assignment', index=False)

        # 8) 可视化
        self.visualize_assignment(tsp_map, df_customers, df_assignment, instance_name)

        used_time = end_time - start_time
        print(f"[TTSLPSolver] 处理完成并保存结果: {output_excel_path}")

        # 9) 记录 summary
        self.record_summary(
            instance_name=instance_name,
            m_customers=m,
            n_candidates=n,
            num_stops=len(selected_stops),
            num_depots=len(selected_depots),
            used_time=used_time
        )

    def visualize_assignment(self, tsp_map, df_customers, df_assignment, instance_name):
        """
        可视化TTSLP的分配结果，并保存图片到data/ttslp
        - 对每个服务点(停靠/甩柜)与其分配到的顾客连虚线
        """
        plt.figure(figsize=(10,10))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 绘制候选服务点(只画被选中的?)
        # 因为df_assignment里只保留了有客户分配的点 -> 选中的点
        # 所以 service_types = [停靠点, 甩柜点]
        service_types = df_assignment['service_type'].unique()
        color_map = {'停靠点':'blue', '甩柜点':'green'}

        for stype in service_types:
            sub = df_assignment[df_assignment['service_type']==stype]
            plt.scatter(
                sub['service_x'], sub['service_y'],
                c=color_map.get(stype,'black'),
                label=stype, alpha=0.6
            )

        # 绘制客户点
        plt.scatter(
            df_customers['x'], df_customers['y'],
            c='red', label='Customers', marker='x'
        )

        # (额外)用虚线把服务点与客户连起来
        for idx, row in df_assignment.iterrows():
            sx, sy = row['service_x'], row['service_y']
            cx, cy = row['customer_x'], row['customer_y']
            plt.plot([sx,cx],[sy,cy],'k--',alpha=0.3)  # 虚线

        plt.title(f"TTSLP Assignment - {instance_name}")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.legend()
        output_path = os.path.join(self.ttslp_output_dir, f"{instance_name}_assignment.png")
        plt.savefig(output_path)
        plt.close()

    def record_summary(self, instance_name, m_customers, n_candidates,
                       num_stops, num_depots, used_time):
        row_dict = {
            'instance_name': instance_name,
            'customers': m_customers,
            'candidates': n_candidates,
            'stops': num_stops,
            'depots': num_depots,
            'used_time': used_time
        }
        df_new = pd.DataFrame([row_dict])
        summary_path = os.path.join(self.ttslp_output_dir, "ttslp_summary.xlsx")

        if os.path.exists(summary_path):
            df_old = pd.read_excel(summary_path)
            df_res = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_res = df_new
        df_res.to_excel(summary_path, index=False)
        print(f"[TTSLPSolver] 已更新 summary => {summary_path}")


if __name__=="__main__":
    config = Config()
    solver = TTSLPSolver(config)

    raw_data_dir = config.DATA_DIR['raw']
    ttslp_output_dir = config.DATA_DIR['ttslp']

    for filename in os.listdir(raw_data_dir):
        if filename.endswith("1.xlsx"):
            instance_path = os.path.join(raw_data_dir, filename)
            # 获取算例名称(不带扩展名)
            instance_name = os.path.splitext(filename)[0]
            # 检查是否已有对应的 xxx_result.xlsx
            result_file = f"{instance_name}_result.xlsx"
            result_path = os.path.join(ttslp_output_dir, result_file)

            if os.path.exists(result_path):
                print(f"[TTSLPSolver] {result_file} 已存在，跳过算例: {instance_path}")
                continue
            else:
                print(f"[TTSLPSolver] 处理算例: {instance_path}")
                solver.solve_ttslp_instance(instance_path)

    print("[TTSLPSolver] 所有算例已处理完成。")
