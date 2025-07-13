import matplotlib.pyplot as plt
import seaborn as sns
import folium
import numpy as np


class Visualizer:
    def __init__(self):
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))

    def plot_instance(self, instance, figsize=(12, 8)):
        """绘制实例的基本情况"""
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制道路
        for _, road in instance['roads'].iterrows():
            ax.plot(
                [road['start_x'], road['end_x']],
                [road['start_y'], road['end_y']],
                'gray', alpha=0.5
            )

        # 绘制客户点
        customers = instance['customers']
        scatter = ax.scatter(
            customers['x'],
            customers['y'],
            c=customers['volume'],
            cmap='YlOrRd',
            s=50,
            alpha=0.6
        )
        plt.colorbar(scatter, label='Demand Volume')

        ax.set_title('Instance Visualization')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')

        return fig

    def plot_solution(self, instance, solution, figsize=(15, 10)):
        """绘制求解结果"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 左图：绘制服务点位置
        self._plot_service_points(instance, solution, ax1)

        # 右图：绘制配送路线
        self._plot_routes(instance, solution, ax2)

        fig.suptitle('Solution Visualization')
        return fig

    def _plot_service_points(self, instance, solution, ax):
        """绘制服务点分布"""
        # 绘制道路
        for _, road in instance['roads'].iterrows():
            ax.plot(
                [road['start_x'], road['end_x']],
                [road['start_y'], road['end_y']],
                'gray', alpha=0.3
            )

        # 绘制停靠点
        ax.scatter(
            solution['selected_stops']['x'],
            solution['selected_stops']['y'],
            c='blue',
            marker='^',
            s=100,
            label='Stops'
        )

        # 绘制甩柜点
        ax.scatter(
            solution['selected_depots']['x'],
            solution['selected_depots']['y'],
            c='red',
            marker='s',
            s=100,
            label='Depots'
        )

        ax.legend()
        ax.set_title('Service Points Distribution')

    def _plot_routes(self, instance, solution, ax):
        """绘制配送路线"""
        # 绘制路线
        for vehicle_id, route in enumerate(solution['routes']):
            color = self.colors[vehicle_id % len(self.colors)]
            route_coords = np.array([
                [point['x'], point['y']] for point in route
            ])

            ax.plot(
                route_coords[:, 0],
                route_coords[:, 1],
                c=color,
                label=f'Vehicle {vehicle_id}'
            )

            # 绘制箭头表示方向
            for i in range(len(route_coords) - 1):
                mid_point = (route_coords[i] + route_coords[i + 1]) / 2
                dx = route_coords[i + 1][0] - route_coords[i][0]
                dy = route_coords[i + 1][1] - route_coords[i][1]
                ax.arrow(
                    mid_point[0], mid_point[1],
                    dx / 10, dy / 10,
                    head_width=0.1,
                    color=color
                )

        ax.legend()
        ax.set_title('Vehicle Routes')

    def create_interactive_map(self, instance, solution):
        """创建交互式地图"""
        # 计算中心点
        center_lat = instance['customers']['y'].mean()
        center_lon = instance['customers']['x'].mean()

        # 创建地图
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13
        )

        # 添加各类点位
        self._add_points_to_map(m, instance, solution)

        # 添加路线
        self._add_routes_to_map(m, solution)

        return m

