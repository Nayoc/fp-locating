import math
import os
import random
from pathlib import Path

import torch

import db.mapper as mapper
from data.optimizer import MaxminNorm
from util.mysql_utils import MySQLConnector

root_dir = str(Path(os.path.abspath(__file__)).parent.parent)
dataset_dir = root_dir + '/data'

min_rsrp = -140


def run(space_id: int, batch_id: str, model='collection'):
    wifi, cell = build_format_dataset(space_id)

    dir_name = '/' + str(space_id) + '/' + model + '_' + batch_id

    directory = os.path.abspath(dataset_dir + dir_name)

    # 如果目录不存在，则创建目录（包括所有上级目录）
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"已创建目录: {directory}")

    space = mapper.select_space(space_id)
    if space is None:
        raise Exception("space is None")

    scale_x = space.get('scale_x')
    scale_rate = space.get('scale_rate')
    scale = math.ceil(scale_x if scale_rate > 1 else scale_x / scale_rate)

    # TODO 暂时只用wifi
    build_tensor_dataset(wifi, scale, directory)

    return dir_name, wifi[0][:-2], cell[0][:-2]


def build_format_dataset(spaceId: int):
    # 方法1: 使用上下文管理器(推荐)
    with MySQLConnector() as db:
        if db.connection.is_connected():
            selectSql = "select * from single_collection_data where space_id = %s"
            results = db.execute_query(selectSql, (spaceId,))

            wifi_header = []
            wifi_data_list = []
            cell_header = []
            cell_data_list = []

            if results:
                for record in results:
                    # 确保记录是字典类型（根据数据库查询结果格式调整）
                    if not isinstance(record, dict):
                        continue

                    # 根据source字段分类
                    if record.get('source') == 'wifi':
                        wifi_data_list.append(record)
                        if not record.get('ap_id') in wifi_header:
                            wifi_header.append(record.get('ap_id'))
                    elif record.get('source') == 'cell':
                        cell_data_list.append(record)
                        if not record.get('ap_id') in cell_header:
                            cell_header.append(record.get('ap_id'))

                wifi_header.append('rp_x')
                wifi_header.append('rp_y')
                cell_header.append('rp_x')
                cell_header.append('rp_y')

                def restructure_data(header_list, data_list, signal_key):
                    """
                    重构单组数据（WiFi或Cell）

                    参数:
                        header_list: 表头
                        data_list: 单组（WiFi或Cell）原始数据列表
                        signal_key: 信号值字段名（'ap_rssi' 或 'ap_rsrp'）

                    返回:
                        重构后的字典列表
                    """
                    # 按(rp_x, rp_y)分组，收集每个坐标点的所有ap信号
                    coordinate_groups = {}

                    for record in data_list:
                        # 获取坐标作为分组键（保留两位小数避免浮点数精度问题）
                        rp_x = round(record.get('rp_x', 0), 2)
                        rp_y = round(record.get('rp_y', 0), 2)
                        coord_key = (rp_x, rp_y)

                        # 获取ap_id和信号值
                        ap_id = record.get('ap_id')
                        signal_value = record.get(signal_key, min_rsrp)  # 使用默认值-999

                        # 初始化该坐标点的字典
                        if coord_key not in coordinate_groups:
                            one_coord_record = [min_rsrp] * (len(header_list) - 2)
                            one_coord_record.append(rp_x)
                            one_coord_record.append(rp_y)
                            one_coord_record[header_list.index(ap_id)] = signal_value
                            coordinate_groups[coord_key] = one_coord_record
                        else:
                            one_coord_record = coordinate_groups[coord_key]
                            one_coord_record[header_list.index(ap_id)] = signal_value

                    # 转换为列表并返回
                    return list(coordinate_groups.values())

                    # 3. 分别处理WiFi和Cell数据
                    # WiFi使用rssi，Cell使用rsrp

                restructured_wifi = restructure_data(wifi_header, wifi_data_list, 'ap_rssi')
                restructured_cell = restructure_data(cell_header, cell_data_list, 'ap_rsrp')

                restructured_wifi.insert(0, wifi_header)
                restructured_cell.insert(0, cell_header)

                return restructured_wifi, restructured_cell


def build_tensor_dataset(dataset: [], scale: float, path):
    dataset = dataset[1:]
    random.shuffle(dataset)

    # 计算分割点（70%的位置）
    split_index = int(len(dataset) * 0.8)

    # 分割为训练集和验证集
    train_set = dataset[:split_index]
    val_set = dataset[split_index:]

    # 提取训练数据和标签（最后两列为标签）
    train_data = [sample[:-2] for sample in train_set]
    train_label = [sample[-2:] for sample in train_set]

    # 提取验证数据和标签（最后两列为标签）
    val_data = [sample[:-2] for sample in val_set]
    val_label = [sample[-2:] for sample in val_set]

    train_data = torch.tensor(train_data)
    train_data = torch.unsqueeze(train_data, dim=1)
    train_label = torch.tensor(train_label)
    print(train_data.shape, train_label.shape)

    val_data = torch.tensor(val_data)
    val_data = torch.unsqueeze(val_data, dim=1)
    val_label = torch.tensor(val_label)
    print(val_data.shape, val_label.shape)

    norm_x = MaxminNorm(max=-30, min=-140)
    norm_y = MaxminNorm(max=scale, min=0)
    # 归一化数据
    train_data = norm_x.norm(train_data)
    val_data = norm_x.norm(val_data)

    train_label = norm_y.norm(train_label)
    val_label = norm_y.norm(val_label)

    torch.save({'data': train_data, 'labels': train_label},
               path + '/train.pth')
    torch.save({'data': val_data, 'labels': val_label},
               path + '/val.pth')


if __name__ == '__main__':
    run(10, '1759800148367')
