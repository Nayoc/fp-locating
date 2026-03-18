import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import db.mapper as mapper
import exec.mtrain as train
import net.mnn as mnn
from util.cdf_utils import CDFPlotter
import numpy as np

root_dir = str(Path(os.path.abspath(__file__)).parent.parent)
data_dir = root_dir + '/data'
cdf_dir = root_dir + '/cdf/'

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


def cdf_draw():
    COLORS = {
        "knn_wifi": "#87CEEB",  # 深蓝色（原KNN WiFi）
        "knn_cell": "#98FB98",  # 浅绿色
        "cnn_wifi": "#FFD166",  # 橙黄色（预留CNN WiFi）
        "cnn_cell": "#FDE68A",  # 砖红色（预留CNN Cell）
        "fusion": "#A23B72"  # 酒红色（KNN Cell）
    }

    plotter = CDFPlotter(figsize=(9, 6), dpi=120)

    knn_wifi_cdf_path = cdf_dir+'knn_wifi_cdf.npz'
    knn_cell_cdf_path = cdf_dir+'knn_cell_cdf.npz'
    cnn_wifi_cdf_path = cdf_dir+'cnn_wifi_cdf.npz'
    cnn_cell_cdf_path = cdf_dir+'cnn_cell_cdf.npz'
    fusion_cdf_path = cdf_dir+'fusion_cdf.npz'

    kd_wifi = np.load(knn_wifi_cdf_path, allow_pickle=True)
    kd_cell = np.load(knn_cell_cdf_path, allow_pickle=True)
    cd_wifi = np.load(cnn_wifi_cdf_path, allow_pickle=True)
    cd_cell = np.load(cnn_cell_cdf_path, allow_pickle=True)
    fd_cell = np.load(fusion_cdf_path, allow_pickle=True)

    val_19 = {
        "cdf_x": [0.0721, 0.2460, 0.2474, 0.3130, 0.3636, 0.4727, 0.4904, 0.5016, 0.5608, 1.0102, 1.0738, 1.4838, 1.5524],
        "cdf_y": [0.0769, 0.1538, 0.2308, 0.3077, 0.3846, 0.4615, 0.5385, 0.6154, 0.6923, 0.7692, 0.8462, 0.9231, 1.0]
    }

    val_15 ={
        "cdf_x": [0.1237, 0.5250, 0.5394, 0.5404, 0.5536, 0.5948, 0.8021, 0.8559, 0.8841, 0.9702, 1.1513, 1.2216, 1.2802, 2.2432],
        "cdf_y": [0.0714, 0.1429, 0.2143, 0.2857, 0.3571, 0.4286, 0.5000, 0.5714, 0.6429, 0.7143, 0.7857, 0.8571, 0.9286, 1.0000]
    }

    # 添加两条CDF曲线（对比两个模型）
    # plotter.add_cdf_curve(kd_wifi["cdf_x"], kd_wifi["cdf_y"], label="KNN_WiFi", color=COLORS["knn_wifi"], linewidth=2)
    # plotter.add_cdf_curve(kd_cell["cdf_x"], kd_cell["cdf_y"], label="KNN_Cell", color=COLORS["knn_cell"], linewidth=2)
    # plotter.add_cdf_curve(cd_wifi["cdf_x"], cd_wifi["cdf_y"], label="CNN_WiFi", color=COLORS["cnn_wifi"], linewidth=2)
    # plotter.add_cdf_curve(cd_cell["cdf_x"], cd_cell["cdf_y"], label="CNN_Cell", color=COLORS["cnn_cell"], linewidth=2)
    # plotter.add_cdf_curve(fd_cell["cdf_x"], fd_cell["cdf_y"], label="CNN_Attention", color=COLORS["fusion"], linewidth=2)
    # plotter.add_cdf_curve(val_19["cdf_x"], val_19["cdf_y"], label="CNN_Attention_实验室验证", color=COLORS["fusion"], linewidth=2)
    plotter.add_cdf_curve(val_15["cdf_x"], val_15["cdf_y"], label="CNN_Attention_居民楼验证", color=COLORS["fusion"], linewidth=2)

    # 标记两个模型的CDF80分位数
    # plotter.mark_quantile(kd_wifi["cdf80"].item(), quantile_percent=80, color=COLORS["knn_wifi"], linestyle="--")
    # plotter.mark_quantile(kd_cell["cdf80"].item(), quantile_percent=80, color=COLORS["knn_cell"], linestyle="--")
    # plotter.mark_quantile(cd_wifi["cdf80"].item(), quantile_percent=80, color=COLORS["cnn_wifi"], linestyle="--")
    # plotter.mark_quantile(cd_cell["cdf80"].item(), quantile_percent=80, color=COLORS["cnn_cell"], linestyle="--")
    # plotter.mark_quantile(fd_cell["cdf80"].item(), quantile_percent=80, color=COLORS["fusion"], linestyle="--")


    # 设置标题和图例
    plotter.set_title("坐标预测误差CDF分布对比")
    plotter.show_legend(loc="lower right")

    # 保存+显示图片
    plotter.show_fig()


if __name__ == '__main__':
    cdf_draw()