import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data as data

import db.mapper as mapper
import exec.mtrain as train
import net.mnn as mnn
from data.optimizer import MaxminNorm

root_dir = str(Path(os.path.abspath(__file__)).parent.parent)
data_dir = root_dir + '/data'


def run(space_id: int, dataset_id: int, data_path, batch_size=50, epochs=300, record_term=30):
    label_norm = None

    file_path = data_dir + data_path
    cell_train_iter, cell_test_iter, wifi_train_iter, wifi_test_iter, fusion_train_iter, fusion_test_iter = load_data(file_path, batch_size)

    # net = mnn.CellWifiFusionModel(cell_in_channels=cell_train_iter.dataset[0][0].shape[0], wifi_in_channels=1, base_feat_channels=5,
    #                             fused_hidden=128, cell_bias_init=1.0)

    net = mnn.BasicCnn()
    loss = nn.MSELoss()

    model_name = str(space_id) + '_' + str(dataset_id)
    model_file = model_name + '.params'
    print('——————————————————————start training——————————————————————')
    mapper.update_dataset_status(dataset_id, 'doing')

    try:
        # train.train(net, fusion_train_iter, fusion_test_iter, loss, epochs, label_norm,
        #             model_file, record_term=record_term)

        train.train(net, wifi_train_iter, wifi_test_iter, loss, epochs, label_norm,
                    model_file, record_term=record_term,mode='single')
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
    return cell_train_iter,cell_test_iter,wifi_train_iter,wifi_test_iter,fusion_train_iter,fusion_test_iter


if __name__ == '__main__':
    run(15, 10012, '/15/collection_1764748570471', batch_size=32)
