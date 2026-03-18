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


def run(space_id: int, dataset_id: int, data_path, batch_size=50, epochs=100, record_term=50):
    print('current train data:' + data_path)
    label_norm = None

    file_path = data_dir + data_path
    cell_train_iter, cell_test_iter, wifi_train_iter, wifi_test_iter, fusion_train_iter, fusion_test_iter,shape = load_data(
        file_path, batch_size)


    # net = mnn.CellWifiFusionModel(cell_in_channels=shape[0],fused_hidden=32)
    # net = mnn.WifiBasicCnn(in_channels=1)
    net = mnn.CellBasicCnn(in_channels=2)
    loss = nn.MSELoss()

    model_name = str(space_id) + '_' + str(dataset_id)
    model_file = model_name + '.params'
    print('——————————————————————start training——————————————————————')
    mapper.update_dataset_status(dataset_id, 'doing')

    try:
        # train.train(net, fusion_train_iter, fusion_test_iter, loss, epochs, label_norm,
        #             model_file, record_term=record_term, mode='multi')

        # train.train(net, wifi_train_iter, wifi_test_iter, loss, epochs, label_norm,
        #             'wifi_'+model_file, record_term=record_term, mode='single')
        #
        train.train(net, cell_train_iter, cell_test_iter, loss, epochs, label_norm,
                    'cell_' + model_file, record_term=record_term, mode='single')
    except Exception as e:
        mapper.update_dataset_status(dataset_id, 'fail')
        raise e
    mapper.update_dataset_status(dataset_id, 'done')
    mapper.update_dataset_model(dataset_id, model_file)
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


if __name__ == '__main__':
    # run(19, 10019, '/19/collection_19test_base1_random', batch_size=128)
    # run(19, 10019, '/19/collection_19test_base2_fix', batch_size=128)
    run(19, 10019, '/19/collection_19test_base2_random', batch_size=128)