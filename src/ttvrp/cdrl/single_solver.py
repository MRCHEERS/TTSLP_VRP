from config.config import Config
from .solver import CDRLClusteringSolver, CDRLRLSolver
import os


def solve_single_case(ttslp_result_file: str):
    """Convenient entry to cluster and run RL on a single TTSLP result."""
    cfg = Config()
    clustering_solver = CDRLClusteringSolver(cfg)
    rl_solver = CDRLRLSolver(cfg)

    if not os.path.exists(ttslp_result_file):
        print(f"[single_solver] 文件不存在: {ttslp_result_file}")
        return

    clustering_solver.run_clustering_single(ttslp_result_file)
    clustered_name = os.path.splitext(os.path.basename(ttslp_result_file))[0] + "_clustered.xlsx"
    rl_solver.run_rl_for_case(clustered_name)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.ttvrp.cdrl.single_solver <ttslp_result.xlsx>")
    else:
        solve_single_case(sys.argv[1])
