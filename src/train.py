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


def run(space_id: int, dataset_id: int, data_path, norm_y, batch_size=50, epochs=300, record_term=30):
    label_norm = MaxminNorm(norm_y, 0)

    file_path = data_dir + data_path
    train_iter, val_iter, shape = load_wifi(file_path, source='cell', batch_size=batch_size)

    # 数组显示8位
    torch.set_printoptions(precision=8)
    net = mnn.CellCNN(in_channels=2)
    loss = nn.CrossEntropyLoss()

    model_name = str(space_id) + '_' + str(dataset_id)
    model_file = model_name + '.params'
    print('——————————————————————start training——————————————————————')
    mapper.update_dataset_status(dataset_id, 'doing')

    try:
        train.train(net, train_iter, val_iter, loss, epochs, label_norm,
                    model_file, record_term=record_term)
    except Exception as e:
        mapper.update_dataset_status(dataset_id, 'fail')
        raise e
    mapper.update_dataset_status(dataset_id, 'done')
    mapper.update_dataset_model(dataset_id, model_file)
    print('——————————————————————ending training——————————————————————')


def load_wifi(data_path, source='cell', batch_size=64):
    train_set = torch.load(data_path + '/' + source + '_train.pth')
    val_set = torch.load(data_path + '/' + source + '_val.pth')

    train_data = train_set['data']
    val_data = val_set['data']

    train_label = train_set['labels']
    val_label = val_set['labels']

    print('训练数据shape:' + str(train_data.shape))

    train_set = data.TensorDataset(train_data, train_label)
    validation_set = data.TensorDataset(val_data, val_label)

    train_iter = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8,
                                 pin_memory=True)
    validation_iter = data.DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=8,
                                      pin_memory=True)

    return train_iter, validation_iter, train_data.shape


if __name__ == '__main__':
    run(11, 10004, '/11/collection_1762914886271', 15, batch_size=50)
