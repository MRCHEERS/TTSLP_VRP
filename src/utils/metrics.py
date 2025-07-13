import numpy as np
from scipy.spatial.distance import cdist


class SolutionEvaluator:
    def __init__(self):
        pass

    def evaluate_solution(self, instance, solution):
        """评估解决方案的各项指标"""
        metrics = {}

        # 计算成本相关指标
        cost_metrics = self._calculate_cost_metrics(instance, solution)
        metrics.update(cost_metrics)

        # 计算服务质量指标
        service_metrics = self._calculate_service_metrics(instance, solution)
        metrics.update(service_metrics)

        # 计算资源利用率指标
        utilization_metrics = self._calculate_utilization_metrics(instance, solution)
        metrics.update(utilization_metrics)

        return metrics

    def _calculate_cost_metrics(self, instance, solution):
        """计算成本相关指标"""
        # 计算总行驶距离
        total_distance = 0
        for route in solution['routes']:
            for i in range(len(route) - 1):
                dist = np.sqrt(
                    (route[i + 1]['x'] - route[i]['x']) ** 2 +
                    (route[i + 1]['y'] - route[i]['y']) ** 2
                )
                total_distance += dist

        # 计算设施成本
        facility_cost = (
                len(solution['selected_stops']) * 100000 +
                len(solution['selected_depots']) * 100000
        )

        return {
            'total_distance': total_distance,
            'facility_cost': facility_cost,
            'total_cost': total_distance + facility_cost
        }

    def _calculate_service_metrics(self, instance, solution):
        """计算服务质量指标"""
        customers = instance['customers']
        stops = solution['selected_stops']
        depots = solution['selected_depots']

        # 计算客户到最近服务点的平均距离
        customer_coords = customers[['x', 'y']].values
        stop_coords = stops[['x', 'y']].values
        depot_coords = depots[['x', 'y']].values

        # 计算距离矩阵
        dist_to_stops = cdist(customer_coords, stop_coords)
        dist_to_depots = cdist(customer_coords, depot_coords)

        # 获取每个客户到最近服务点的距离
        min_distances = np.minimum(
            dist_to_stops.min(axis=1),
            dist_to_depots.min(axis=1)
        )

        return {
            'avg_service_distance': min_distances.mean(),
            'max_service_distance': min_distances.max(),
            'service_distance_std': min_distances.std()
        }

    def _calculate_utilization_metrics(self, instance, solution):
        """计算资源利用率指标"""
        # 计算车辆利用率
        vehicle_loads = []
        vehicle_capacity = instance['metadata'].get('vehicle_capacity', 500)

        for route in solution['routes']:
            route_load = sum(point.get('volume', 0) for point in route)
            vehicle_loads.append(route_load)

        # 计算设施利用率
        stop_loads = {}
        depot_loads = {}

        for route in solution['routes']:
            for point in route:
                if point['type'] == 'stop':
                    stop_loads[point['id']] = stop_loads.get(point['id'], 0) + point['volume']
                elif point['type'] == 'depot':
                    depot_loads[point['id']] = depot_loads.get(point['id'], 0) + point['volume']

        return {
            'avg_vehicle_utilization': np.mean(vehicle_loads) / vehicle_capacity,
            'avg_stop_utilization': np.mean(list(stop_loads.values())) / 120,  # theta
            'avg_depot_utilization': np.mean(list(depot_loads.values())) / 480  # rho
        }

    def generate_report(self, metrics):
        """生成评估报告"""
        report = "Solution Evaluation Report\n"
        report += "========================\n\n"

        # 成本分析
        report += "Cost Analysis:\n"
        report += f"- Total Distance: {metrics['total_distance']:.2f}\n"
        report += f"- Facility Cost: {metrics['facility_cost']:.2f}\n"
        report += f"- Total Cost: {metrics['total_cost']:.2f}\n\n"

        # 服务质量分析
        report += "Service Quality Analysis:\n"
        report += f"- Average Service Distance: {metrics['avg_service_distance']:.2f}\n"
        report += f"- Maximum Service Distance: {metrics['max_service_distance']:.2f}\n"
        report += f"- Service Distance Std: {metrics['service_distance_std']:.2f}\n\n"

        # 资源利用率分析
        report += "Resource Utilization Analysis:\n"
        report += f"- Average Vehicle Utilization: {metrics['avg_vehicle_utilization']:.2%}\n"
        report += f"- Average Stop Utilization: {metrics['avg_stop_utilization']:.2%}\n"
        report += f"- Average Depot Utilization: {metrics['avg_depot_utilization']:.2%}\n"

        return report
