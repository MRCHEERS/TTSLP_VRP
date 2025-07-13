# ttslp_vrp/config/config.py

import os

class Config:
    # 实验规模与对应的地图尺寸
    EXPERIMENT_SCALES = {
        'S': [500, 1000],    # 小规模
        'M': [1500, 2000],   # 中规模
        'L': [2500, 3000]    # 大规模
    }

    # 顾客数量
    CUSTOMER_NUMS = [10, 20, 30, 40, 50, 60]

    # 需求范围
    DEMAND_RANGES = {
        'L': (10, 30),   # Low demand
        'M': (50, 100),  # Medium demand
        'H': (80, 150)   # High demand
    }

    # TTSLP参数
    TTSLP_PARAMS = {
        'R_t': 100,   # 停靠点最大服务半径
        'R_d': 120,   # 甩柜点最大服务半径
        'theta': 120, # 停靠点密度阈值
        'rho': 480    # 甩柜点密度阈值
    }

    # TTVRP参数
    TTVRP_PARAMS = {
        'vehicle_capacity': 500, # 车辆容量
        'num_vehicles': 4        # 车辆数量
    }

    # CDRL参数
    CDRL_PARAMS = {
        'alpha': 0.5,         # 学习率
        'gamma': 0.01,        # 折扣因子
        'epsilon': 0.5,       # 初始探索率
        'final_epsilon': 0.05,# 最终探索率
        'iterations': 10000   # 训练迭代次数
    }

    # 算例生成参数
    INSTANCE_PARAMS = {
        'cases_per_combination': 5  # 每组组合生成的算例数量
    }

    # 文件命名规则
    FILENAME_FORMAT = "{scale}-{map_size}-{customer_num}-{demand_symbol}-{case_id}.xlsx"

    # =========== 下面是规范化的数据目录配置 ===========
    # 先获取 config.py 所在目录（与 data/、src/ 同级）
    _ROOT = os.path.dirname(os.path.abspath(__file__))  # .../ttslp_vrp/config

    # 项目根目录（向上一层）
    _PROJECT_ROOT = os.path.abspath(os.path.join(_ROOT, '..'))  # .../ttslp_vrp

    # data 目录
    _DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')  # .../ttslp_vrp/data

    DATA_DIR = {
        # 例如: {PROJECT_ROOT}/data/raw
        'raw': os.path.join(_DATA_DIR, 'raw2'),
        # 例如: {PROJECT_ROOT}/data/processed
        'processed': os.path.join(_DATA_DIR, 'processed'),
        # 例如: {PROJECT_ROOT}/data/ttslp
        'ttslp': os.path.join(_DATA_DIR, 'ttslp2'),
        # 如果需要更多路径，如:
        'ttvrp': os.path.join(_DATA_DIR, 'ttvrp2'),
        # 'gurobi': os.path.join(_DATA_DIR, 'gurobi'),
    }