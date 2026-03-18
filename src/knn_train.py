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


def knn_run(space_id: int, dataset_id: int, data_path, batch_size=50):
    print('current train data:' + data_path)

    file_path = data_dir + data_path
    cell_train,cell_test,wifi_train,wifi_test = load_knn_data(
        file_path, batch_size)

    knn = mnn.MKNN()
    print('——————————————————————start training——————————————————————')

    try:
        knn.fit(cell_train[0],cell_train[1])
        accuracy, mean_error, cdf80 = knn.predict(cell_test[0],cell_test[1])

        # knn.fit(wifi_train[0], wifi_train[1])
        # accuracy, mean_error, cdf80 = knn.predict(wifi_test[0], wifi_test[1])

        print(accuracy, mean_error, cdf80)


    except Exception as e:
        raise e
    print('——————————————————————ending training——————————————————————')

def load_knn_data(path, batch_size=64):
    cell = torch.load(path + '/cell.pth')
    wifi = torch.load(path + '/wifi.pth')
    fusion = torch.load(path + '/fusion.pth')

    return cell['train'].tensors,cell['test'].tensors,wifi['train'].tensors,wifi['test'].tensors


if __name__ == '__main__':
    knn_run(19, 10016, '/19/collection_19test_base2_random', batch_size=10000)
