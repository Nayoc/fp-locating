import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from data.dataset import FusionDataset
from util.mysql_utils import MySQLConnector
from util.txt_utils import TxtArrayTool

root_dir = str(Path(os.path.abspath(__file__)).parent.parent)
dataset_dir = root_dir + '/data'


def run(space_id: int, batch_id: str, model='collection'):
    dir_name = '/' + str(space_id) + '/' + model + '_' + batch_id

    directory = os.path.abspath(dataset_dir + dir_name)

    # 如果目录不存在，则创建目录（包括所有上级目录）
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"已创建目录: {directory}")

    scale = 15

    wifi_set, cell_set = split_train_test(space_id)

    cell_data = build_cell_format_dataset_multi_channel(cell_set, directory)
    wifi_data = build_wifi_format_dataset(wifi_set, directory)
    build_tensor_dataset(cell_data, wifi_data, path=directory)

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

                # random.shuffle(coordinate_groups)
                #
                # split_index = int(len(coordinate_groups) * 0.8)
                #
                # train_coord_set = coordinate_groups[:split_index]

                wifi_set = []
                # wifi_test_set = []
                cell_set = []
                # cell_test_set = []

                for row in results:
                    if row.get('source') == 'wifi':
                        wifi_set.append(row)

                    elif row.get('source') == 'cell':

                        cell_set.append(row)

    return wifi_set, cell_set,


# ---------------------- 辅助函数：时序滑动截取（不变） ----------------------
def slide_extend_pic(data, step, slide_step):
    t_len = len(data)
    if t_len < step:
        pad_len = step - t_len
        pad_data = np.pad(data, ((0, pad_len), (0, 0)), mode='constant', constant_values=data[-1:])
        return np.expand_dims(pad_data, axis=0)
    else:
        n_frag = (t_len - step) // slide_step + 1
        frags = [data[i * slide_step: i * slide_step + step] for i in range(n_frag)]
        return np.array(frags, dtype=np.float32)


def build_cell_format_dataset_multi_channel(dataset, path, step=16):
    cell_order_file = os.path.join(path, 'cell_order.txt')
    text_tool = TxtArrayTool()
    cell_order = []

    if os.path.exists(cell_order_file):
        cell_order = text_tool.read(cell_order_file)
        cell_order = list(dict.fromkeys(cell_order))
    else:
        unique_ap_ids = []
        for row in dataset:
            ap_id = row.get('ap_id')
            if ap_id and ap_id not in unique_ap_ids:
                unique_ap_ids.append(ap_id)
        cell_order = unique_ap_ids

        text_tool.write(cell_order_file, cell_order)

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

        # 取所有有数据基站的原始信号长度，用最大值作为默认信号的长度（确保滑动后片段数一致）
        valid_signal_lengths = []
        for ap_id in ap_dict:
            signal_len = len(ap_dict[ap_id])
            if signal_len > 0:
                valid_signal_lengths.append(signal_len)

        # 若当前坐标下所有基站都无数据，默认信号长度设为 step（避免空列表）
        default_signal_len = max(valid_signal_lengths) if valid_signal_lengths else step

        # ---------------------- 按cell_order顺序遍历基站（核心修改） ----------------------
        for ap_id in cell_order:
            if ap_id not in ap_dict:
                default_signal = [[-120.0, -20.0, -10.0]] * default_signal_len  # 长度=default_signal_len
                result = slide_extend_pic(default_signal, step, slide_step)
            else:
                signal_data = ap_dict[ap_id]
                result = slide_extend_pic(signal_data, step, slide_step)
            all_fingerprints.append(result)

        # ---------------------- 该坐标下所有片段合并（按基站顺序+滑动顺序） ----------------------
        if all_fingerprints:  # 仅保留有有效片段的坐标
            final_coord_data[coord_key] = all_fingerprints

    return final_coord_data


def build_wifi_format_dataset(dataset, path, step=16, max_ap=10):
    # 1.固定wifi顺序
    text_tool = TxtArrayTool()
    header = text_tool.read(path + '/wifi_order.txt')

    if not header:
        for record in dataset:
            ap_id = record.get('ap_id')
            if ap_id and ap_id not in header and len(header) < max_ap:
                header.append(ap_id)
        # 若AP不足10个，剩余位置用无效标记填充（后续用-120dBm填充）
        while len(header) < max_ap:
            header.append(f"AP_EMPTY_{len(header)}")

        text_tool.write(path + '/wifi_order.txt', header)

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
            coordinate_groups[coord_key][batch_id] = [-120] * len(header)

        if ap_id in header:
            coordinate_groups[coord_key][batch_id][header.index(ap_id)] = rssi

    coordinate_groups = {
        coord: list(sub_dict.values()) for coord, sub_dict in coordinate_groups.items()
    }

    slide_step = step // 2
    final_coord_data = {}
    for coord_key, signal_data in coordinate_groups.items():
        result = slide_extend_pic(signal_data, step, slide_step)
        final_coord_data[coord_key] = result

    return final_coord_data


def build_tensor_dataset(cell_data, wifi_data, path=None, test_ratio=0.3, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---------------------- 0. 收集所有唯一坐标并划分训练/测试坐标 ----------------------
    # 收集所有存在的坐标（Cell和WiFi的坐标全集）
    all_coords = list(set(cell_data.keys()).union(set(wifi_data.keys())))
    print(f"总坐标点数：{len(all_coords)}")

    # 对坐标进行随机划分（核心修改：先划分坐标，再按坐标收集样本）
    np.random.shuffle(all_coords)
    test_coord_size = int(len(all_coords) * test_ratio)
    train_coords = set(all_coords[test_coord_size:])  # 训练坐标集合
    test_coords = set(all_coords[:test_coord_size])  # 测试坐标集合
    print(f"训练坐标数：{len(train_coords)}, 测试坐标数：{len(test_coords)}")

    # ---------------------- 1. 处理Cell数据（按划分后的坐标收集样本） ----------------------
    cell_train_samples, cell_train_labels = [], []
    cell_test_samples, cell_test_labels = [], []
    m = None  # 基站数（所有坐标统一）

    # 先确定统一的基站数m（取第一个有效坐标的基站数）
    for coord in all_coords:
        if coord in cell_data and len(cell_data[coord]) > 0:
            m = len(cell_data[coord])
            break
    if m is None:
        raise ValueError("Cell数据中无有效基站信息")

    for coord, cell_frags in cell_data.items():
        # 过滤基站数不统一的坐标
        if len(cell_frags) != m:
            continue

        # Cell内部按基站最小指纹数对齐（保证m个基站样本数一致）
        ap_frag_lens = [len(frags) for frags in cell_frags]
        min_ap_len = min(ap_frag_lens) if ap_frag_lens else 0
        if min_ap_len == 0:
            continue

        # 构建当前坐标的所有Cell样本
        cell_batch = np.swapaxes(cell_frags, 0, 1)
        cell_batch_labels = [coord] * min_ap_len

        # 根据坐标划分，分配到训练集或测试集
        if coord in train_coords:
            cell_train_samples.extend(cell_batch)
            cell_train_labels.extend(cell_batch_labels)
        elif coord in test_coords:
            cell_test_samples.extend(cell_batch)
            cell_test_labels.extend(cell_batch_labels)

    # 转换为TensorDataset
    cell_train = TensorDataset(
        torch.tensor(np.array(cell_train_samples, dtype=np.float32)),
        torch.tensor(np.array(cell_train_labels, dtype=np.float32))
    )
    cell_test = TensorDataset(
        torch.tensor(np.array(cell_test_samples, dtype=np.float32)),
        torch.tensor(np.array(cell_test_labels, dtype=np.float32))
    )
    print(f"Cell - 训练集{len(cell_train)}样本, 测试集{len(cell_test)}样本 (基站数m：{m})")

    # ---------------------- 2. 处理WiFi数据（按划分后的坐标收集样本） ----------------------
    wifi_train_samples, wifi_train_labels = [], []
    wifi_test_samples, wifi_test_labels = [], []

    for coord, wifi_frags in wifi_data.items():
        wifi_frags_np = np.array(wifi_frags, dtype=np.float32)  # (n_w, 16, 20)
        if len(wifi_frags_np) == 0:
            continue

        # 构建当前坐标的所有WiFi样本
        wifi_batch = wifi_frags_np[:, None, ...]  # (n_w, 1, 16, 20)
        wifi_batch_labels = [coord] * len(wifi_frags_np)

        # 根据坐标划分，分配到训练集或测试集
        if coord in train_coords:
            wifi_train_samples.extend(wifi_batch)
            wifi_train_labels.extend(wifi_batch_labels)
        elif coord in test_coords:
            wifi_test_samples.extend(wifi_batch)
            wifi_test_labels.extend(wifi_batch_labels)

    # 转换为TensorDataset
    wifi_train = TensorDataset(
        torch.tensor(np.array(wifi_train_samples, dtype=np.float32)),
        torch.tensor(np.array(wifi_train_labels, dtype=np.float32))
    )
    wifi_test = TensorDataset(
        torch.tensor(np.array(wifi_test_samples, dtype=np.float32)),
        torch.tensor(np.array(wifi_test_labels, dtype=np.float32))
    )
    print(f"WiFi - 训练集{len(wifi_train)}样本, 测试集{len(wifi_test)}样本")

    # ---------------------- 3. 融合数据集（按划分后的坐标收集样本，保持Cell/WiFi对应） ----------------------
    def build_fusion_dataset(coords):
        """根据坐标集合构建融合数据集"""
        fusion_cell, fusion_wifi, fusion_labels = [], [], []

        for coord in coords:
            # 处理当前坐标的Cell样本
            cell_samps = []
            if coord in cell_data and len(cell_data[coord]) == m:
                cell_frags = cell_data[coord]
                ap_frag_lens = [len(frags) for frags in cell_frags]
                min_ap_len = min(ap_frag_lens) if ap_frag_lens else 0
                if min_ap_len > 0:
                    cell_samps = np.stack([frags[:min_ap_len] for frags in cell_frags], axis=1)  # (n_c, m, 16, 3)

            # 处理当前坐标的WiFi样本
            wifi_samps = []
            if coord in wifi_data:
                wifi_frags_np = np.array(wifi_data[coord])
                if len(wifi_frags_np) > 0:
                    wifi_samps = wifi_frags_np[:, None, ...]  # (n_w, 1, 16, 20)

            # 处理样本数不一致的情况：用0填充缺失的模态数据
            max_samples = max(len(cell_samps), len(wifi_samps))

            if max_samples > 0:
                # 填充Cell数据（不足则用0填充）
                if len(cell_samps) < max_samples:
                    cell_pad = np.zeros((max_samples - len(cell_samps), m, 16, 3), dtype=np.float32)
                    cell_full = np.concatenate([cell_samps, cell_pad], axis=0)
                else:
                    cell_full = cell_samps[:max_samples]  # 防止超出（理论上不会）

                # 填充WiFi数据（不足则用0填充）
                if len(wifi_samps) < max_samples:
                    wifi_pad = np.zeros((max_samples - len(wifi_samps), 1, 16, 10), dtype=np.float32)
                    wifi_full = np.concatenate([wifi_samps, wifi_pad], axis=0)
                else:
                    wifi_full = wifi_samps[:max_samples]  # 防止超出（理论上不会）

                # 添加到融合数据集
                fusion_cell.extend(cell_full)
                fusion_wifi.extend(wifi_full)
                fusion_labels.extend([coord] * max_samples)

        return FusionDataset(
            np.array(fusion_cell, dtype=np.float32),
            np.array(fusion_wifi, dtype=np.float32),
            np.array(fusion_labels, dtype=np.float32)
        )

    # 构建融合训练集和测试集
    fusion_train = build_fusion_dataset(train_coords)
    fusion_test = build_fusion_dataset(test_coords)
    print(f"Fusion - 训练集{len(fusion_train)}样本, 测试集{len(fusion_test)}样本")

    # ---------------------- 4. 保存pth文件 ----------------------
    if path:
        torch.save({'train': cell_train, 'test': cell_test}, path + '/cell.pth')
        torch.save({'train': wifi_train, 'test': wifi_test}, path + '/wifi.pth')
        torch.save({'train': fusion_train, 'test': fusion_test}, path + '/fusion.pth')
        print("\n数据集保存完成：")
    print(f"- Cell: 训练集{len(cell_train)}样本, 测试集{len(cell_test)}样本 (shape: {cell_train[0][0].shape[1:]})")
    print(f"- WiFi: 训练集{len(wifi_train)}样本, 测试集{len(wifi_test)}样本 (shape: {wifi_train[0][0].shape[1:]})")
    print(f"- Fusion: 训练集{len(fusion_train)}样本, 测试集{len(fusion_test)}样本")


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
    run(10, '1764498718525')
