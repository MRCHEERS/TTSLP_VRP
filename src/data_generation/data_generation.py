# ttslp_vrp/src/data_generation/generator.py

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config.config import Config  # 使用绝对导入路径

class DataGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.raw_data_dir = config.DATA_DIR['raw']
        self.processed_data_dir = config.DATA_DIR['processed']
        self.ensure_directories()

    def ensure_directories(self):
        """确保数据存储目录存在"""
        for dir_path in [self.raw_data_dir, self.processed_data_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"[DataGenerator] 创建目录: {dir_path}")

    def generate_random_instance(self, map_size, num_customers, demand_symbol, demand_range, output_path):
        """
        生成单个随机算例并保存到指定路径
        """
        # 生成客户名字
        customer_names = [f"C{i+1}" for i in range(num_customers)]

        # 生成随机坐标
        customer_x = np.random.randint(0, map_size, num_customers)
        customer_y = np.random.randint(0, map_size, num_customers)

        # 生成需求量
        customer_demands = np.random.randint(demand_range[0], demand_range[1]+1, num_customers)

        # 组装 DataFrame
        df_customers = pd.DataFrame({
            "name": customer_names,
            "x": customer_x,
            "y": customer_y,
            "volume": customer_demands
        })

        # 生成道路
        road_interval = 100  # 每隔 road_interval 一条水平/垂直路
        horizontal_roads = []
        for i in range(0, map_size, road_interval):
            horizontal_roads.append([f"Horizontal Road{i}", 'West End', 0, i])
            horizontal_roads.append([f"Horizontal Road{i}", 'East End', map_size, i])

        vertical_roads = []
        for i in range(0, map_size, road_interval):
            vertical_roads.append([f"Vertical Road{i}", 'South End', i, 0])
            vertical_roads.append([f"Vertical Road{i}", 'North End', i, map_size])

        df_roads = pd.DataFrame(horizontal_roads + vertical_roads,
                                columns=['Road_Name', 'Sub_Name', 'x', 'y'])

        # 将 df_customer 与 df_roads 写入Excel
        with pd.ExcelWriter(output_path) as writer:
            df_customers.to_excel(writer, sheet_name='customer', index=False)
            df_roads.to_excel(writer, sheet_name='road', index=False)

        # 可选：简单可视化
        plt.figure(figsize=(8, 8))
        plt.scatter(df_customers['x'], df_customers['y'], c='blue', label='Customers')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Instance: {os.path.basename(output_path)}")
        plt.legend()
        plt.savefig(os.path.splitext(output_path)[0] + '.png')
        plt.close()

        print(f"[DataGenerator] 生成并保存算例: {output_path}")

    def run_generation(self):
        """
        根据配置生成所有算例
        """
        scales = self.config.EXPERIMENT_SCALES  # {'S': [500, 1000], ...}
        customer_nums = self.config.CUSTOMER_NUMS  # [10, 20, ..., 60]
        demand_ranges = self.config.DEMAND_RANGES  # {'L':(10,30), ...}
        cases_per_combination = self.config.INSTANCE_PARAMS['cases_per_combination']
        filename_format = self.config.FILENAME_FORMAT

        for scale, map_sizes in scales.items():
            for map_size in map_sizes:
                for customer_num in customer_nums:
                    for demand_symbol, demand_range in demand_ranges.items():
                        for case_id in range(1, cases_per_combination + 1):
                            # 构建文件名
                            filename = filename_format.format(
                                scale=scale,
                                map_size=map_size,
                                customer_num=customer_num,
                                demand_symbol=demand_symbol,
                                case_id=case_id
                            )
                            output_path = os.path.join(self.raw_data_dir, filename)

                            # 检查文件是否已存在，避免重复生成
                            if os.path.exists(output_path):
                                print(f"[DataGenerator] 文件已存在，跳过: {output_path}")
                                continue

                            # 生成并保存算例
                            self.generate_random_instance(
                                map_size=map_size,
                                num_customers=customer_num,
                                demand_symbol=demand_symbol,
                                demand_range=demand_range,
                                output_path=output_path
                            )

if __name__ == "__main__":
    # 确保在项目根目录下运行此脚本
    config = Config()
    generator = DataGenerator(config)
    generator.run_generation()
