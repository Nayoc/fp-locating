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
model_dir = root_dir + '/models'


def run(space_id: int, dataset_id: int, data_path, norm_y, batch_size=128, epochs=100, record_term=500):
    file_path = data_dir + data_path
    train_iter, val_iter, shape = load(file_path, batch_size=batch_size)

    # 数组显示8位
    torch.set_printoptions(precision=8)
    net = mnn.MFCNN1D()
    loss = nn.CrossEntropyLoss()

    label_norm = MaxminNorm(norm_y, 0)

    model_name = str(space_id) + '_' + str(dataset_id)
    model_file = model_dir + '/' + model_name + '.params'
    print('——————————————————————start training——————————————————————')
    mapper.update_train_status(dataset_id, 'doing')

    try:
        train.train(net, train_iter, val_iter, loss, epochs, label_norm,
                    model_file=model_file, record_term=record_term)
    except Exception as e:
        mapper.update_train_status(dataset_id, 'fail')
        raise e
    print('——————————————————————ending training——————————————————————')
    mapper.update_train_status(dataset_id, 'done')


def load(data_path, batch_size=64):
    train = torch.load(data_path + '/train.pth')
    val = torch.load(data_path + '/val.pth')

    train_data = train['data']
    val_data = val['data']

    train_label = train['labels']
    val_label = val['labels']

    train_set = data.TensorDataset(train_data, train_label)
    validation_set = data.TensorDataset(val_data, val_label)

    train_iter = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8,
                                 pin_memory=True)
    validation_iter = data.DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=8,
                                      pin_memory=True)

    return train_iter, validation_iter, train_data.shape


def update_train_result():
    mapper.select_fingerprint_dataset()


if __name__ == '__main__':
    run(6, 1, 14, batch_size=128, epochs=300, record_term=50)
