"""
卡尔曼滤波信号处理工具
实现一维信号的卡尔曼滤波算法，并提供可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import time
from util.mysql_utils import MySQLConnector
from datetime import datetime
import random

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

class KalmanFilter:
    """卡尔曼滤波器类"""

    def __init__(self, q=0.01, r=1.0, p=1.0, x_initial=None):
        """
        初始化卡尔曼滤波器

        参数:
            q: 过程噪声协方差
            r: 测量噪声协方差
            p: 初始估计误差协方差
            x_initial: 初始状态估计值，如果为None则使用第一个测量值
        """
        self.q = q  # 过程噪声协方差
        self.r = r  # 测量噪声协方差
        self.p = p  # 估计误差协方差
        self.x = x_initial  # 状态估计
        self.k = 0  # 卡尔曼增益

    def update(self, measurement):
        """
        使用卡尔曼滤波算法更新状态估计

        参数:
            measurement: 当前测量值

        返回:
            滤波后的估计值
        """
        # 如果是第一次更新，初始化状态估计
        if self.x is None:
            self.x = measurement
            return self.x

        # 预测步骤
        # 状态预测：x_pred = x_prev (假设系统模型为x_k = x_{k-1})
        x_pred = self.x
        # 误差协方差预测：p_pred = p_prev + q
        p_pred = self.p + self.q

        # 更新步骤
        # 计算卡尔曼增益：k = p_pred / (p_pred + r)
        self.k = p_pred / (p_pred + self.r)
        # 更新状态估计：x = x_pred + k * (z - x_pred)
        self.x = x_pred + self.k * (measurement - x_pred)
        # 更新误差协方差：p = (1 - k) * p_pred
        self.p = (1 - self.k) * p_pred

        return self.x


def generate_sample_data(n_points=100, base_value=-90, noise_level=10, trend_strength=0.1):
    """
    生成示例信号数据

    参数:
        n_points: 数据点数量
        base_value: 基础信号值
        noise_level: 噪声水平
        trend_strength: 趋势强度

    返回:
        生成的信号数据列表
    """
    # 生成基础信号（带轻微趋势）
    trend = np.linspace(0, trend_strength * n_points, n_points)
    # 生成随机噪声
    noise = np.random.normal(0, noise_level, n_points)
    # 组合信号
    signal = base_value + trend + noise
    # 确保信号在-60到-120范围内
    signal = np.clip(signal, -120, -60)

    return signal.tolist()


class KalmanFilterVisualizer:
    """卡尔曼滤波可视化工具类"""

    def __init__(self, data=None, q=0.05, r=1.0, p=1.,ymin=-110,ymax=-95):
        """
        初始化可视化工具

        参数:
            data: 输入信号数据，如果为None则生成示例数据
            q: 过程噪声协方差
            r: 测量噪声协方差
            p: 初始估计误差协方差
        """
        # 如果没有提供数据，生成示例数据
        self.data = data if data is not None else generate_sample_data()
        self.filtered_data = []
        self.q = q
        self.r = r
        self.p = p
        self.filter = KalmanFilter(q, r, p)

        # 创建图形和子图
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        # 调整布局以留出空间给滑块和按钮
        plt.subplots_adjust(bottom=0.25, right=0.85)

        # 初始化图表
        self.original_line, = self.ax.plot(self.data, 'ro', markersize=5, label='原始信号')
        self.filtered_line, = self.ax.plot([], 'g-', linewidth=2, label='滤波后信号')

        # 设置图表属性
        self.ax.set_title('RSSI过滤结果')
        self.ax.set_xlabel('索引序号')
        self.ax.set_ylabel('信号值')
        self.ax.set_ylim(ymin, ymax)  # 设置y轴范围略大于信号范围
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right')

        # 添加参数滑块
        # self._add_sliders()
        #
        # # 添加按钮
        # self._add_buttons()

        # 初始化滤波结果
        self.update_filter()

    def _add_sliders(self):
        """添加参数调整滑块"""
        # Q参数滑块
        ax_q = plt.axes([0.2, 0.15, 0.65, 0.03])
        self.slider_q = Slider(
            ax=ax_q,
            label='过程噪声协方差 Q',
            valmin=0.001,
            valmax=0.1,
            valinit=self.q,
            valstep=0.001
        )
        self.slider_q.on_changed(self.update_q)

        # R参数滑块
        ax_r = plt.axes([0.2, 0.1, 0.65, 0.03])
        self.slider_r = Slider(
            ax=ax_r,
            label='测量噪声协方差 R',
            valmin=0.1,
            valmax=10.0,
            valinit=self.r,
            valstep=0.1
        )
        self.slider_r.on_changed(self.update_r)

        # P参数滑块
        ax_p = plt.axes([0.2, 0.05, 0.65, 0.03])
        self.slider_p = Slider(
            ax=ax_p,
            label='初始估计误差协方差 P',
            valmin=0.1,
            valmax=10.0,
            valinit=self.p,
            valstep=0.1
        )
        self.slider_p.on_changed(self.update_p)

    def _add_buttons(self):
        """添加功能按钮"""
        # 重置按钮
        ax_reset = plt.axes([0.87, 0.15, 0.1, 0.04])
        self.button_reset = Button(ax_reset, '重置')
        self.button_reset.on_clicked(self.reset_filter)

        # 保存按钮
        ax_save = plt.axes([0.87, 0.1, 0.1, 0.04])
        self.button_save = Button(ax_save, '保存结果')
        self.button_save.on_clicked(self.save_results)

        # 生成新数据按钮
        ax_new_data = plt.axes([0.87, 0.05, 0.1, 0.04])
        self.button_new_data = Button(ax_new_data, '新数据')
        self.button_new_data.on_clicked(self.generate_new_data)

    def update_q(self, val):
        """更新Q参数"""
        self.q = val
        self.update_filter()

    def update_r(self, val):
        """更新R参数"""
        self.r = val
        self.update_filter()

    def update_p(self, val):
        """更新P参数"""
        self.p = val
        self.update_filter()

    def update_filter(self):
        """使用当前参数更新滤波结果"""
        # 重置滤波器
        self.filter = KalmanFilter(self.q, self.r, self.p)
        # 应用滤波
        self.filtered_data = [self.filter.update(measurement) for measurement in self.data]
        # 更新图表
        self.filtered_line.set_data(range(len(self.filtered_data)), self.filtered_data)
        # 刷新图表
        self.fig.canvas.draw_idle()

    def get_filter_data(self):
        return self.filtered_data

    def reset_filter(self, event):
        """重置滤波器参数到默认值"""
        self.q = 0.01
        self.r = 1.0
        self.p = 1.0

        # 更新滑块位置
        # self.slider_q.set_val(self.q)
        # self.slider_r.set_val(self.r)
        # self.slider_p.set_val(self.p)

        # 更新滤波结果
        self.update_filter()

    def save_results(self, event):
        """保存滤波结果和图表"""
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存图表
        plt.figure(self.fig.number)
        plt.savefig(f'kalman_filter_result_{timestamp}.png', dpi=300, bbox_inches='tight')

        # 保存数据到CSV文件
        np.savetxt(
            f'kalman_filter_data_{timestamp}.csv',
            np.column_stack((self.data, self.filtered_data)),
            delimiter=',',
            header='原始数据,滤波后数据',
            comments=''
        )

        print(f"结果已保存为 'kalman_filter_result_{timestamp}.png' 和 'kalman_filter_data_{timestamp}.csv'")

    def generate_new_data(self, event):
        """生成新的随机数据"""
        self.data = generate_sample_data()
        self.original_line.set_data(range(len(self.data)), self.data)
        self.update_filter()

    def show(self):
        """显示可视化界面"""
        plt.show()

    def print_results(self):
        """打印滤波结果的前10个值"""
        print("原始数据前10个值:")
        print(self.data[:10])
        print("\n滤波后数据前10个值:")
        print(self.filtered_data[:10])

def build_single_data(space_id,rp_x,rp_y,ap_id):
    with MySQLConnector() as db:
        if db.connection.is_connected():
            # 1. 执行Cell和WiFi的分组查询
            sql = """select ap_rsrp from single_collection_data where space_id=%s and rp_x=%s and rp_y= %s and ap_id=%s"""
            results = db.execute_query(sql,(space_id,rp_x,rp_y,ap_id))
            rsrp_list = [item['ap_rsrp'] for item in results]
            return rsrp_list


def process_signal_data(data_list):
    """
    信号数据处理主方法：
    1. 给每个整数添加0-1之间的两位随机小数
    2. 随机挑选部分值，减去10-20之间的两位随机浮点数
    3. 打乱最终数据列表
    :param data_list: 原始整数列表
    :return: 处理后的乱序列表
    """
    # 第一步：给每个值添加0-1的两位随机小数
    processed_list = []
    for num in data_list:
        random_decimal_0_1 = round(random.uniform(0, 1), 2)
        new_num = num + random_decimal_0_1
        processed_list.append(new_num)

    # 第二步：随机挑选部分值，减去10-20之间的两位随机浮点数
    # 随机确定要修改的数量（1~列表长度的一半，避免修改过多）
    # modify_count = random.randint(1, max(1,len(processed_list)))
    # modify_count = len(processed_list)//2
    # modify_count = 50
    # # 随机选择要修改的索引（不重复）
    # modify_indices = random.sample(range(len(processed_list)), modify_count)
    #
    # for idx in modify_indices:
    #     random_decimal_10_20 = round(random.uniform(0, 2), 2)
    #
    #     mode = random.choice((1, 0))
    #     if mode==1:
    #         processed_list[idx] = round(processed_list[idx] - random_decimal_10_20,2)
    #     else:
    #         processed_list[idx] = round(processed_list[idx] + random_decimal_10_20,2)

    return processed_list


def calculate_variance(num_list, is_sample=True):
    """
    计算数值列表的方差
    :param num_list: 待计算的数值列表（如processed_list）
    :param is_sample: 是否计算样本方差（True=样本方差，False=总体方差，默认False）
    :return: 方差值（保留4位小数）
    """
    # 空列表校验
    if len(num_list) == 0:
        raise ValueError("列表不能为空，无法计算方差")
    if len(num_list) == 1 and is_sample:
        raise ValueError("样本方差要求列表长度至少为2")

    # 步骤1：计算列表的均值
    mean = round(sum(num_list) / len(num_list),2)

    # 步骤2：计算每个值与均值的差的平方和
    squared_diff_sum = sum((x - mean) ** 2 for x in num_list)

    # 步骤3：计算方差（总体方差/样本方差）
    if is_sample:
        # 样本方差：除以 n-1（n为列表长度）
        variance = squared_diff_sum / (len(num_list) - 1)
    else:
        # 总体方差：除以 n
        variance = squared_diff_sum / len(num_list)

    # 保留4位小数，避免精度冗余
    return round(variance, 4)

if __name__ == "__main__":
    # 生成示例数据
    # data = generate_sample_data(n_points=100, base_value=-90, noise_level=10)
    # rsrp_list = build_single_data(19, 1.02, 6.04,"a4:a9:30:c6:88:6e")
    rsrp_list = build_single_data(19, 1.02, 6.04,16022)
    # processed_list = process_signal_data(rsrp_list)
    processed_list=rsrp_list
    print("方差："+str(calculate_variance(processed_list)))

    ymax = max(processed_list)+1
    ymin = min(processed_list)-1
    # 创建并显示可视化工具
    visualizer = KalmanFilterVisualizer(processed_list,ymin=ymin,ymax=ymax)

    # 打印结果
    # visualizer.print_results()

    print("方差："+str(calculate_variance(visualizer.get_filter_data())))
    # 显示图形界面
    # visualizer.show()