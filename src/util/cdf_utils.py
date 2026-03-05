import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# 全局配置（可选：解决中文显示问题，按需启用）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

class CDFPlotter:
    """
    CDF绘图工具类（x轴：误差/m，y轴：累积概率0-1）
    支持：
    1. 绘制单/多条CDF曲线（对比不同模型）
    2. 标记指定分位数（如CDF80/90）
    3. 标记准确率阈值
    4. 保存图片/显示图片
    """

    def __init__(self, figsize=(8, 6), dpi=100):
        """
        初始化绘图工具
        :param figsize: 图片尺寸 (宽, 高)
        :param dpi: 图片分辨率
        """
        self.figsize = figsize
        self.dpi = dpi
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        # 初始化样式
        self.ax.set_xlabel("误差/m", fontsize=12)
        self.ax.set_ylabel("累积概率", fontsize=12)
        self.ax.set_ylim(0, 0.9)  # y轴固定0-1.05（留出余量）
        self.ax.grid(alpha=0.3, linestyle='-')
        self.ax.tick_params(axis='both', labelsize=10)

    def _convert_to_numpy(self, data):
        """统一转换为numpy数组（兼容torch张量）"""
        if isinstance(data, torch.Tensor):
            return data.cpu().detach().numpy()
        elif isinstance(data, list):
            return np.array(data)
        return data

    def add_cdf_curve(self, cdf_x, cdf_y, label="CDF曲线", color="blue", linewidth=2, linestyle="-"):
        """
        添加一条CDF曲线
        :param cdf_x: 排序后的误差值（x轴），支持torch/numpy/list
        :param cdf_y: 对应的累积概率（y轴），支持torch/numpy/list
        :param label: 曲线标签（图例显示）
        :param color: 曲线颜色
        :param linewidth: 线宽
        :param linestyle: 线型（-/:--/-.等）
        """
        # 数据格式统一
        cdf_x = self._convert_to_numpy(cdf_x)
        cdf_y = self._convert_to_numpy(cdf_y)

        # 绘制CDF曲线
        self.ax.plot(
            cdf_x, cdf_y,
            label=label,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle
        )

    def mark_quantile(self, quantile_value, quantile_percent=80, color="red", linestyle="--"):
        """
        标记指定分位数（如CDF80）
        :param quantile_value: 分位数对应的误差值（如cdf80）
        :param quantile_percent: 分位数百分比（如80/90）
        :param color: 标记线颜色
        :param linestyle: 标记线线型
        """
        quantile_value = self._convert_to_numpy(quantile_value)
        # 绘制垂直分位线（x轴）
        self.ax.axvline(
            x=quantile_value,
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            label=f"CDF{quantile_percent}: {quantile_value:.4f} m"
        )
        # 绘制水平概率线（y轴）
        self.ax.axhline(
            y=quantile_percent / 100,
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            alpha=0.5  # 半透明避免遮挡
        )

    def mark_threshold(self, threshold, label="准确率阈值", color="green", linestyle=":"):
        """
        标记误差阈值（如判定准确的误差上限）
        :param threshold: 误差阈值（m）
        :param label: 阈值标签
        :param color: 颜色
        :param linestyle: 线型
        """
        self.ax.axvline(
            x=threshold,
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            label=f"{label}: {threshold} m"
        )

    def set_title(self, title, fontsize=14):
        """设置图表标题"""
        self.ax.set_title(title, fontsize=fontsize, pad=10)

    def show_legend(self, loc="lower right"):
        """显示图例"""
        self.ax.legend(loc=loc, fontsize=10, framealpha=0.9)

    def save_fig(self, save_path, bbox_inches="tight", pad_inches=0.2):
        """
        保存图片
        :param save_path: 保存路径（如"cdf_plot.png"）
        :param bbox_inches: 紧凑布局
        :param pad_inches: 边距
        """
        self.fig.savefig(
            save_path,
            dpi=self.dpi,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches
        )
        print(f"CDF图已保存至：{save_path}")

    def show_fig(self):
        """显示图片"""
        plt.tight_layout()
        plt.show()

    def close_fig(self):
        """关闭画布（释放内存）"""
        plt.close(self.fig)


