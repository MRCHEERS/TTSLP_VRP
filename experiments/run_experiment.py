import os
import json
import time
import pandas as pd
from datetime import datetime
from pathlib import Path

from src.data_generation.generator import InstanceGenerator
from src.ttslp.solver import TTSLPSolver
from src.ttvrp.solver import TTVRPSolver
from src.utils.visualization import Visualizer
from src.utils.metrics import SolutionEvaluator
from config.config import Config


class ExperimentRunner:
    def __init__(self, config_path=None):
        self.config = Config()
        self.instance_generator = InstanceGenerator(self.config)
        self.visualizer = Visualizer()
        self.evaluator = SolutionEvaluator()

        # 创建实验目录
        self.experiment_dir = self._create_experiment_dir()

    def run_full_experiment(self):
        """运行完整的实验流程"""
        results = []

        # 遍历所有实验场景
        for scale in self.config.EXPERIMENT_SCALES.keys():
            for customer_num in self.config.CUSTOMER_NUMS:
                for demand_type in self.config.DEMAND_RANGES.keys():
                    result = self._run_single_experiment(
                        scale,
                        customer_num,
                        demand_type
                    )
                    results.append(result)

        # 保存实验结果
        self._save_results(results)

        return results

    def _run_single_experiment(self, scale, customer_num, demand_type):
        """运行单个实验场景"""
        print(f"\nRunning experiment: {scale}-{customer_num}-{demand_type}")

        # 生成实验实例
        instance = self.instance_generator.generate_instance(
            scale,
            customer_num,
            demand_type
        )

        # 求解TTSLP
        ttslp_solver = TTSLPSolver(self.config.TTSLP_PARAMS)
        ttslp_start = time.time()
        ttslp_solution = ttslp_solver.solve(instance)
        ttslp_time = time.time() - ttslp_start

        # 求解TTVRP
        ttvrp_solver = TTVRPSolver(
            method='cdrl',
            params=self.config.TTVRP_PARAMS
        )
        ttvrp_start = time.time()
        ttvrp_solution = ttvrp_solver.solve(ttslp_solution)
        ttvrp_time = time.time() - ttvrp_start

        # 评估解决方案
        metrics = self.evaluator.evaluate_solution(instance, ttvrp_solution)

        # 生成可视化
        self._generate_visualizations(
            instance,
            ttslp_solution,
            ttvrp_solution,
            f"{scale}_{customer_num}_{demand_type}"
        )

        return {
            'scenario': {
                'scale': scale,
                'customer_num': customer_num,
                'demand_type': demand_type
            },
            'times': {
                'ttslp_time': ttslp_time,
                'ttvrp_time': ttvrp_time
            },
            'metrics': metrics
        }

    def _create_experiment_dir(self):
        """创建实验目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = Path(f"results/experiment_{timestamp}")

        # 创建必要的子目录
        for subdir in ['visualizations', 'data', 'logs']:
            (experiment_dir / subdir).mkdir(parents=True, exist_ok=True)

        return experiment_dir

    def _generate_visualizations(self, instance, ttslp_sol, ttvrp_sol, name):
        """生成并保存可视化结果"""
        # 实例可视化
        instance_fig = self.visualizer.plot_instance(instance)
        instance_fig.savefig(
            self.experiment_dir / 'visualizations' / f'{name}_instance.png'
        )

        # TTSLP解决方案可视化
        ttslp_fig = self.visualizer.plot_solution(instance, ttslp_sol)
        ttslp_fig.savefig(
            self.experiment_dir / 'visualizations' / f'{name}_ttslp.png'
        )

        # TTVRP解决方案可视化
        ttvrp_fig = self.visualizer.plot_solution(instance, ttvrp_sol)
        ttvrp_fig.savefig(
            self.experiment_dir / 'visualizations' / f'{name}_ttvrp.png'
        )

        # 创建交互式地图
        interactive_map = self.visualizer.create_interactive_map(
            instance,
            ttvrp_sol
        )
        interactive_map.save(
            self.experiment_dir / 'visualizations' / f'{name}_map.html'
        )

    def _save_results(self, results):
        """保存实验结果"""
        # 保存详细结果
        with open(self.experiment_dir / 'data' / 'detailed_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        # 创建汇总表格
        summary_data = []
        for result in results:
            row = {
                **result['scenario'],
                **result['times'],
                **{f"metric_{k}": v for k, v in result['metrics'].items()}
            }
            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(
            self.experiment_dir / 'data' / 'summary_results.csv',
            index=False
        )


if __name__ == "__main__":
    runner = ExperimentRunner()
    results = runner.run_full_experiment()
    print("Experiment completed successfully!")
