import os
import random
from pathlib import Path

import numpy as np
import torch

from data.optimizer import MaxminNorm
from util.mysql_utils import MySQLConnector
from util.txt_utils import TxtArrayTool

root_dir = str(Path(os.path.abspath(__file__)).parent.parent)
dataset_dir = root_dir + '/data'

min_rsrp = -140


def run(space_id: int, batch_id: str, model='collection'):
    dir_name = '/' + str(space_id) + '/' + model + '_' + batch_id

    directory = os.path.abspath(dataset_dir + dir_name)

    # 如果目录不存在，则创建目录（包括所有上级目录）
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"已创建目录: {directory}")

    scale = 15

    wifi_train_set, wifi_test_set, cell_train_set, cell_test_set = split_train_test(space_id)

    build_tensor_dataset(build_cell_format_dataset_3_channel(cell_train_set, step=5), directory, source='cell',
                         set_type='train')
    build_tensor_dataset(build_cell_format_dataset_3_channel(cell_test_set, step=5), directory, source='cell',
                         set_type='val')
    build_tensor_dataset(build_wifi_format_dataset(wifi_train_set, directory, step=5), directory, source='wifi',
                         set_type='train')
    build_tensor_dataset(build_wifi_format_dataset(wifi_test_set, directory, step=5), directory, source='wifi',
                         set_type='val')

    return dir_name


def split_train_test(spaceId: int):
    with MySQLConnector() as db:
        if db.connection.is_connected():
            selectSql = "select * from single_collection_data where space_id = %s order by request_batch_id asc"
            results = db.execute_query(selectSql, (spaceId,))

            coordinate_groups = []

            if results:
                # ---------------------- 步骤1：按ap_id分组（每个ap_id对应一个基站的所有数据） ----------------------
                for row in results:
                    # 获取坐标作为分组键（保留两位小数避免浮点数精度问题）
                    rp_x = round(row.get('rp_x', 0), 2)
                    rp_y = round(row.get('rp_y', 0), 2)
                    coord_key = (rp_x, rp_y)

                    if coord_key not in coordinate_groups:
                        coordinate_groups.append(coord_key)

                random.shuffle(coordinate_groups)

                split_index = int(len(coordinate_groups) * 0.8)

                train_coord_set = coordinate_groups[:split_index]

                wifi_train_set = []
                wifi_test_set = []
                cell_train_set = []
                cell_test_set = []

                for row in results:
                    if row.get('source') == 'wifi':
                        if (row.get('rp_x', 0), row.get('rp_y', 0)) in train_coord_set:
                            wifi_train_set.append(row)
                        else:
                            wifi_test_set.append(row)
                    elif row.get('source') == 'cell':
                        if (row.get('rp_x', 0), row.get('rp_y', 0)) in train_coord_set:
                            cell_train_set.append(row)
                        else:
                            cell_test_set.append(row)

    return wifi_train_set, wifi_test_set, cell_train_set, cell_test_set


# 不同的基站分为不同的通道
def build_cell_format_dataset_3_channel(dataset, step=16):
    coordinate_groups = {}

    # ---------------------- 步骤1：按ap_id分组（每个ap_id对应一个基站的所有数据） ----------------------
    for row in dataset:
        # 获取坐标作为分组键（保留两位小数避免浮点数精度问题）
        rp_x = round(row.get('rp_x', 0), 2)
        rp_y = round(row.get('rp_y', 0), 2)
        coord_key = (rp_x, rp_y)

        ap_id = row.get('ap_id')

        rsrp = float(row.get('ap_rsrp', -120))  # 无效值用-120填充（符合基站信号范围）
        rsrq = float(row.get('ap_rsrq', -20))  # 无效值用-20填充
        sinr = float(row.get('ap_sinr', -10))  # 无效值用-10填充

        if coord_key not in coordinate_groups:
            coordinate_groups[coord_key] = {}

        if ap_id not in coordinate_groups[coord_key]:
            coordinate_groups[coord_key][ap_id] = []

        coordinate_groups[coord_key][ap_id].append([rsrp, rsrq, sinr])

    final_coord_data = {}
    slide_step = step // 2

    for (rp_x, rp_y), ap_dict in coordinate_groups.items():
        coord_key = (rp_x, rp_y)
        all_fingerprints = []  # 存储该坐标下所有基站的所有滑动片段

        # ---------------------- 遍历该坐标下的每个基站 ----------------------
        for ap_id, signal_data in ap_dict.items():
            result = slide_extend_pic(signal_data, step, slide_step)
            all_fingerprints.append(result)

        # ---------------------- 该坐标下所有片段合并（按基站顺序+滑动顺序） ----------------------
        if all_fingerprints:  # 仅保留有有效片段的坐标
            final_coord_data[coord_key] = all_fingerprints

    train_samples = []
    for (x, y), fingerprints in final_coord_data.items():
        # 标签：坐标 (x,y) 转换为 shape=(2,) 的数组
        label = np.array([x, y], dtype=np.float32)

        data = np.array(fingerprints, dtype=np.float32)
        data = data.swapaxes(0, 1)
        # 遍历该坐标下的所有基站指纹片段
        for fp in data:
            # 添加到训练样本列表（确保格式符合 (3,step), (2,)）
            train_samples.append((fp, label))

    return train_samples


# TODO 基站没有再分出一个维度，后续根据训练效果再调整
# def build_cell_format_dataset_1_channel(dataset, step=16):
#     coordinate_groups = {}
#
#     # ---------------------- 步骤1：按ap_id分组（每个ap_id对应一个基站的所有数据） ----------------------
#     for row in dataset:
#         # 获取坐标作为分组键（保留两位小数避免浮点数精度问题）
#         rp_x = round(row.get('rp_x', 0), 2)
#         rp_y = round(row.get('rp_y', 0), 2)
#         coord_key = (rp_x, rp_y)
#
#         ap_id = row.get('ap_id')
#
#         rsrp = float(row.get('ap_rsrp', -120))  # 无效值用-120填充（符合基站信号范围）
#         rsrq = float(row.get('ap_rsrq', -20))  # 无效值用-20填充
#         sinr = float(row.get('ap_sinr', -10))  # 无效值用-10填充
#         rp_x = float(row.get('rp_x', 0))
#         rp_y = float(row.get('rp_y', 0))
#
#         if coord_key not in coordinate_groups:
#             coordinate_groups[coord_key] = {}
#
#         if ap_id not in coordinate_groups[coord_key]:
#             coordinate_groups[coord_key][ap_id] = []
#
#         coordinate_groups[coord_key][ap_id].append([rsrp, rsrq, sinr])
#
#     # ---------------------- 步骤2：对每个基站的数据按step裁剪（生成固定长度时序片段） ----------------------
#     final_coord_data = {}
#     slide_step = step // 2
#
#     # ---------------------- 遍历每个坐标（x,y） ----------------------
#     for (rp_x, rp_y), ap_dict in coordinate_groups.items():
#         coord_key = (rp_x, rp_y)
#         all_fingerprints = []  # 存储该坐标下所有基站的所有滑动片段
#
#         # ---------------------- 遍历该坐标下的每个基站 ----------------------
#         for ap_id, signal_data in ap_dict.items():
#             result = slide_extend_pic(signal_data, step, slide_step)
#             all_fingerprints.extend(result)
#
#         # ---------------------- 该坐标下所有片段合并（按基站顺序+滑动顺序） ----------------------
#         if all_fingerprints:  # 仅保留有有效片段的坐标
#             final_coord_data[coord_key] = all_fingerprints
#
#     train_samples = []
#     for (x, y), fingerprints in final_coord_data.items():
#         # 标签：坐标 (x,y) 转换为 shape=(2,) 的数组
#         label = np.array([x, y], dtype=np.float32)  # 对应 (1*2,) 要求
#
#         # 遍历该坐标下的所有基站指纹片段
#         for fp in fingerprints:
#             # 添加到训练样本列表（确保格式符合 (3,step), (2,)）
#             train_samples.append((fp, label))
#
#     return train_samples


def build_wifi_format_dataset(dataset, path, step=16, max_ap=20):
    # 1.固定wifi顺序
    header = []
    for record in dataset:
        ap_id = record.get('ap_id')
        if ap_id and ap_id not in header and len(header) < max_ap:
            header.append(ap_id)
    # 若AP不足20个，剩余位置用无效标记填充（后续用-120dBm填充）
    while len(header) < max_ap:
        header.append(f"AP_EMPTY_{len(header)}")

    # 2.根据rp点分组数据
    coordinate_groups = {}
    for row in dataset:
        # 获取坐标作为分组键（保留两位小数避免浮点数精度问题）
        rp_x = round(row.get('rp_x', 0), 2)
        rp_y = round(row.get('rp_y', 0), 2)
        coord_key = (rp_x, rp_y)

        ap_id = row.get('ap_id')
        rssi = float(row.get('ap_rssi', -120))  # 无效值用-120填充（符合基站信号范围）
        batch_id = row.get('request_batch_id')

        if coord_key not in coordinate_groups:
            coordinate_groups[coord_key] = {}

        if batch_id not in coordinate_groups[coord_key]:
            coordinate_groups[coord_key][batch_id] = [] * max_ap

        coordinate_groups[coord_key][batch_id].insert(header.index(ap_id), rssi)

    coordinate_groups = {
        coord: list(sub_dict.values()) for coord, sub_dict in coordinate_groups.items()
    }

    slide_step = step // 2
    final_coord_data = {}
    for coord_key, signal_data in coordinate_groups.items():
        result = slide_extend_pic(signal_data, step, slide_step)
        final_coord_data[coord_key] = result

    train_samples = []
    for (x, y), fingerprints in final_coord_data.items():
        # 标签：坐标 (x,y) 转换为 shape=(2,) 的数组
        label = np.array([x, y], dtype=np.float32)  # 对应 (1*2,) 要求

        # 遍历该坐标下的所有基站指纹片段
        for fp in fingerprints:
            # 添加到训练样本列表（确保格式符合 (3,step), (2,)）
            train_samples.append((fp, label))

    text_tool = TxtArrayTool()
    text_tool.write(path + '/header_order.txt', header)

    return train_samples


def build_tensor_dataset(dataset: [], path, scale=None, source='cell', set_type='train'):
    file = '/' + source + '_' + set_type + '.pth'

    # 提取训练数据和标签（最后两列为标签）
    data = [sample[0] for sample in dataset]
    label = [sample[1] for sample in dataset]

    data = torch.tensor(data)

    if len(data.shape) < 4:
        data = torch.unsqueeze(data, dim=1)
    label = torch.tensor(label)
    print(data.shape, label.shape)

    # 归一化数据
    if scale:
        norm_x = MaxminNorm(max=-30, min=-140)
        norm_y = MaxminNorm(max=scale, min=0)
        data = norm_x.norm(data)
        label = norm_y.norm(label)

    torch.save({'data': data, 'labels': label}, path + file)


def slide_extend_pic(dataset, step: int, slide_step: int):
    result = []

    time_steps = len(dataset)
    if time_steps < step:
        result.append(dataset)
    else:
        for i in range(0, time_steps - step + 1, slide_step):
            window_data = dataset[i:i + step]
            window_arr = np.array(window_data, dtype=np.float32)
            result.append(window_arr)

    return result


if __name__ == '__main__':
    run(11, '1762914886271')
