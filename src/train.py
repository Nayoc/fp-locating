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

plt.rcParams['font.sans-serif'] = ['SimHei']

def run(space_id: int, dataset_id: int, data_path, batch_size=50, epochs=300, record_term=50):
    print('current train data:' + data_path)
    label_norm = None

    file_path = data_dir + data_path
    cell_train_iter, cell_test_iter, wifi_train_iter, wifi_test_iter, fusion_train_iter, fusion_test_iter,shape = load_data(
        file_path, batch_size)

    # net = mnn.CellWifiFusionModel(cell_in_channels=cell_train_iter.dataset[0][0].shape[0], wifi_in_channels=1, base_feat_channels=5,
    #                             fused_hidden=128, cell_bias_init=1.0)

    net = mnn.CellWifiFusionModel(cell_in_channels=shape[0],fused_hidden=32)
    # net = mnn.WifiBasicCnn(in_channels=1)
    # net = mnn.CellBasicCnn(in_channels=1)
    loss = nn.MSELoss()

    model_name = str(space_id) + '_' + str(dataset_id)
    model_file = model_name + '.params'
    print('——————————————————————start training——————————————————————')
    mapper.update_dataset_status(dataset_id, 'doing')

    try:
        train.train(net, fusion_train_iter, fusion_test_iter, loss, epochs, label_norm,
                    model_file, record_term=record_term, mode='multi')

        # train.train(net, wifi_train_iter, wifi_test_iter, loss, epochs, label_norm,
        #             'wifi_'+model_file, record_term=record_term, mode='single')
        #
        # train.train(net, cell_train_iter, cell_test_iter, loss, epochs, label_norm,
        #             'cell_' + model_file, record_term=record_term, mode='single')
    except Exception as e:
        mapper.update_dataset_status(dataset_id, 'fail')
        raise e
    mapper.update_dataset_status(dataset_id, 'done')
    mapper.update_dataset_model(dataset_id, model_file)
    print('——————————————————————ending training——————————————————————')



def knn_run(space_id: int, dataset_id: int, data_path, batch_size=50):
    print('current train data:' + data_path)

    file_path = data_dir + data_path
    cell_train,cell_test,wifi_train,wifi_test = load_knn_data(
        file_path, batch_size)

    knn = mnn.MKNN()
    print('——————————————————————start training——————————————————————')

    try:
        # knn.fit(cell_train[0],cell_train[1])
        # accuracy, mean_error, cdf80 = knn.predict(cell_test[0],cell_test[1])

        knn.fit(wifi_train[0], wifi_train[1])
        accuracy, mean_error, cdf80 = knn.predict(wifi_test[0], wifi_test[1])

        print(accuracy, mean_error, cdf80)


    except Exception as e:
        raise e
    print('——————————————————————ending training——————————————————————')


def load_data(path, batch_size=64):
    cell = torch.load(path + '/cell.pth')
    wifi = torch.load(path + '/wifi.pth')
    fusion = torch.load(path + '/fusion.pth')

    cell_train_iter = data.DataLoader(cell['train'], batch_size=batch_size, shuffle=True, num_workers=8,
                                      pin_memory=True)
    cell_test_iter = data.DataLoader(cell['test'], batch_size=batch_size, shuffle=True, num_workers=8,
                                     pin_memory=True)

    wifi_train_iter = data.DataLoader(wifi['train'], batch_size=batch_size, shuffle=True, num_workers=8,
                                      pin_memory=True)
    wifi_test_iter = data.DataLoader(wifi['test'], batch_size=batch_size, shuffle=True, num_workers=8,
                                     pin_memory=True)

    fusion_train_iter = data.DataLoader(fusion['train'], batch_size=batch_size, shuffle=True, num_workers=8,
                                        pin_memory=True)
    fusion_test_iter = data.DataLoader(fusion['test'], batch_size=batch_size, shuffle=True, num_workers=8,
                                       pin_memory=True)

    print(fusion['train'][0][0].shape)
    return cell_train_iter, cell_test_iter, wifi_train_iter, wifi_test_iter, fusion_train_iter, fusion_test_iter,fusion['train'][0][0].shape

def load_knn_data(path, batch_size=64):
    cell = torch.load(path + '/cell.pth')
    wifi = torch.load(path + '/wifi.pth')
    fusion = torch.load(path + '/fusion.pth')

    return cell['train'].tensors,cell['test'].tensors,wifi['train'].tensors,wifi['test'].tensors

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

    # 添加两条CDF曲线（对比两个模型）
    plotter.add_cdf_curve(kd_wifi["cdf_x"], kd_wifi["cdf_y"], label="KNN_WiFi", color=COLORS["knn_wifi"], linewidth=2)
    plotter.add_cdf_curve(kd_cell["cdf_x"], kd_cell["cdf_y"], label="KNN_Cell", color=COLORS["knn_cell"], linewidth=2)
    plotter.add_cdf_curve(cd_wifi["cdf_x"], cd_wifi["cdf_y"], label="CNN_WiFi", color=COLORS["cnn_wifi"], linewidth=2)
    plotter.add_cdf_curve(cd_cell["cdf_x"], cd_cell["cdf_y"], label="CNN_Cell", color=COLORS["cnn_cell"], linewidth=2)
    plotter.add_cdf_curve(fd_cell["cdf_x"], fd_cell["cdf_y"], label="CNN_Attention", color=COLORS["fusion"], linewidth=2)

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
    # run(19, 10019, '/19/collection_19test_02', batch_size=32)
    # knn_run(19, 10016, '/19/collection_19test_02', batch_size=10000)
    cdf_draw()