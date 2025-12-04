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


# 最小集合
# def build_cell_format_dataset_multi_channel(dataset, path, step=16):
#     cell_order_file = os.path.join(path, 'cell_order.txt')
#     text_tool = TxtArrayTool()
#     cell_order = []
#
#     # ---------------------- 核心修改1：先构建坐标分组，再筛选全局公共基站 ----------------------
#     coordinate_groups = {}
#     for row in dataset:
#         rp_x = round(row.get('rp_x', 0), 2)
#         rp_y = round(row.get('rp_y', 0), 2)
#         coord_key = (rp_x, rp_y)
#         ap_id = row.get('ap_id')
#
#         if not ap_id:  # 跳过无AP ID的无效数据
#             continue
#
#         rsrp = float(row.get('ap_rsrp', -120))
#         rsrq = float(row.get('ap_rsrq', -20))
#         sinr = float(row.get('ap_sinr', -10))
#
#         if coord_key not in coordinate_groups:
#             coordinate_groups[coord_key] = {}
#         if ap_id not in coordinate_groups[coord_key]:
#             coordinate_groups[coord_key][ap_id] = []
#         coordinate_groups[coord_key][ap_id].append([rsrp, rsrq, sinr])
#
#     # 步骤2：统计每个基站的“有效坐标数”（即该基站在多少个坐标点上有数据）
#     total_coord_count = len(coordinate_groups)  # 总坐标点数量
#     ap_coord_count = {}  # key: ap_id, value: 该基站存在的坐标数
#     for coord_key, ap_dict in coordinate_groups.items():
#         for ap_id in ap_dict:
#             if len(ap_dict[ap_id]) > 0:  # 仅统计有有效信号的基站
#                 ap_coord_count[ap_id] = ap_coord_count.get(ap_id, 0) + 1
#
#     # 步骤3：筛选“全局公共基站”（存在于所有坐标点的基站）
#     common_ap_ids = [ap_id for ap_id, count in ap_coord_count.items() if count == total_coord_count]
#     if not common_ap_ids:
#         raise ValueError("无全局公共基站！所有基站都未在所有坐标点上出现，请检查数据集。")
#     print(f"全局公共基站数：{len(common_ap_ids)}，基站列表：{common_ap_ids}")
#
#     # ---------------------- 基站顺序读取/保存（基于公共基站） ----------------------
#     if os.path.exists(cell_order_file):
#         # 读取已保存的顺序，仅保留其中的公共基站（确保兼容性）
#         saved_order = text_tool.read(cell_order_file)
#         cell_order = [ap_id for ap_id in saved_order if ap_id in common_ap_ids]
#         # 补充未在saved_order中但属于公共基站的ID（按首次出现顺序）
#         for ap_id in common_ap_ids:
#             if ap_id not in cell_order:
#                 cell_order.append(ap_id)
#     else:
#         # 首次运行：公共基站按“数据集首次出现顺序”排序
#         unique_common_aps = []
#         for row in dataset:
#             ap_id = row.get('ap_id')
#             if ap_id in common_ap_ids and ap_id not in unique_common_aps:
#                 unique_common_aps.append(ap_id)
#         cell_order = unique_common_aps
#         # 保存公共基站顺序（覆盖原cell_order.txt，仅存公共基站）
#         text_tool.write(cell_order_file, cell_order)
#         print(f"公共基站顺序已保存到：{cell_order_file}")
#
#     # ---------------------- 数据构建（仅用公共基站，无默认填充） ----------------------
#     final_coord_data = {}
#     slide_step = step // 2
#
#     for (rp_x, rp_y), ap_dict in coordinate_groups.items():
#         coord_key = (rp_x, rp_y)
#         all_fingerprints = []
#
#         # 按公共基站顺序遍历（每个基站在当前坐标都有数据，无需默认填充）
#         for ap_id in cell_order:
#             signal_data = ap_dict[ap_id]  # 必然存在（公共基站特性）
#             result = slide_extend_pic(signal_data, step, slide_step)
#             all_fingerprints.append(result)
#
#         # 验证所有基站的片段数一致（可选，调试用）
#         frag_counts = [len(frag) for frag in all_fingerprints]
#         if len(set(frag_counts)) > 1:
#             print(f"警告：坐标({rp_x},{rp_y})的基站片段数不一致：{frag_counts}")
#
#         final_coord_data[coord_key] = all_fingerprints
#
#     return final_coord_data

# 填充默认值
def build_cell_format_dataset_multi_channel(dataset, save_path, step=16):
    cell_order_file = os.path.join(save_path, 'cell_order.txt')
    text_tool = TxtArrayTool()
    cell_order = []

    # 读取/生成基站顺序（原逻辑不变）
    if os.path.exists(cell_order_file):
        print(f"读取已存在的基站顺序：{cell_order_file}")
        cell_order = text_tool.read(cell_order_file)
        cell_order = list(dict.fromkeys(cell_order))
    else:
        print("未找到基站顺序文件，将自动生成并保存")
        unique_ap_ids = []
        for row in dataset:
            ap_id = row.get('ap_id')
            if ap_id and ap_id not in unique_ap_ids:
                unique_ap_ids.append(ap_id)
        cell_order = unique_ap_ids
        text_tool.write(cell_order_file, cell_order)
        print(f"基站顺序已保存到：{cell_order_file}")

    coordinate_groups = {}

    # 按坐标+AP分组（原逻辑不变）
    for row in dataset:
        rp_x = round(row.get('rp_x', 0), 2)
        rp_y = round(row.get('rp_y', 0), 2)
        coord_key = (rp_x, rp_y)
        ap_id = row.get('ap_id')
        rsrp = float(row.get('ap_rsrp', -120))
        rsrq = float(row.get('ap_rsrq', -20))
        sinr = float(row.get('ap_sinr', -10))

        if coord_key not in coordinate_groups:
            coordinate_groups[coord_key] = {}
        if ap_id not in coordinate_groups[coord_key]:
            coordinate_groups[coord_key][ap_id] = []
        coordinate_groups[coord_key][ap_id].append([rsrp, rsrq, sinr])

    final_coord_data = {}
    slide_step = step // 2

    for (rp_x, rp_y), ap_dict in coordinate_groups.items():
        coord_key = (rp_x, rp_y)
        all_fingerprints = []

        # ---------------------- 核心调整：滑动前统一所有基站的原始信号时序长度（复制补全） ----------------------
        # 步骤1：计算当前坐标下所有有数据基站的原始信号最大时序长度
        valid_signal_lengths = []
        for ap_id in ap_dict:
            signal_len = len(ap_dict[ap_id])
            if signal_len > 0:
                valid_signal_lengths.append(signal_len)
        # 若当前坐标无有效基站数据，默认最大长度设为step（避免滑动后片段维度异常）
        max_signal_len = max(valid_signal_lengths) if valid_signal_lengths else step
        print(f"坐标({rp_x},{rp_y})：滑动前统一时序长度为 {max_signal_len}")

        # 步骤2：按统一长度补全原始信号（复制补全，保持列表格式适配原滑动函数）
        for ap_id in cell_order:
            if ap_id not in ap_dict:
                # 情况1：无基站数据 → 生成默认信号（[-120,-20,-10]），复制补全到max_signal_len
                base_default = [[-120.0, -20.0, -10.0]]  # 基础默认信号（列表格式）
                if max_signal_len > 1:
                    # 循环复制基础默认信号（列表乘法），截取到max_signal_len
                    repeat_times = (max_signal_len // 1) + 1
                    completed_signal = (base_default * repeat_times)[:max_signal_len]
                else:
                    completed_signal = base_default
            else:
                # 情况2：有基站数据 → 提取原始信号（列表格式），复制补全到max_signal_len
                raw_signal = ap_dict[ap_id]  # 原数据是列表，直接使用
                raw_len = len(raw_signal)
                if raw_len == 0:
                    # 原始信号为空，按无数据处理
                    base_default = [[-120.0, -20.0, -10.0]]
                    completed_signal = (base_default * max_signal_len)[:max_signal_len]
                elif raw_len < max_signal_len:
                    # 时序长度不足 → 复制自身补全（列表乘法实现循环复制）
                    repeat_times = (max_signal_len // raw_len) + 1
                    completed_signal = (raw_signal * repeat_times)[:max_signal_len]
                else:
                    # 时序长度足够 → 截取前max_signal_len个，避免长度超标
                    completed_signal = raw_signal[:max_signal_len]

            # ---------------------- 调用原 slide_extend_pic 函数（完全未修改） ----------------------
            frags = slide_extend_pic(completed_signal, step, slide_step)

            # 原滑动函数在time_steps≥step时返回(step,3)，不足时返回原长度（需强制补全到step）
            for idx, frag in enumerate(frags):
                frag_arr = np.array(frag, dtype=np.float32)
                if frag_arr.shape != (step, 3):
                    # 片段长度不足step，用最后一个元素复制补全到step
                    fill_num = step - frag_arr.shape[0]
                    if fill_num > 0:
                        last_element = frag_arr[-1:]  # 取最后一个时序点（保持(1,3)）
                        fill_elements = (last_element.tolist() * fill_num)[:fill_num]  # 复制补全
                        frag_completed = frag + fill_elements  # 列表拼接
                        frags[idx] = np.array(frag_completed, dtype=np.float32)
            # 转换所有片段为numpy数组，确保输出格式统一
            frags_np = np.array(frags, dtype=np.float32)

            # 最终校验：滑动+补全后片段维度必须是 (num_frags, step, 3)
            assert frags_np.shape[1:] == (step, 3), f"基站{ap_id}最终片段维度错误：{frags_np.shape[1:]}≠({step},3)"
            all_fingerprints.append(frags_np)

        # 验证所有基站的指纹图数量一致（避免后续堆叠报错）
        frag_counts = [len(frag) for frag in all_fingerprints]
        assert len(set(frag_counts)) == 1, f"坐标({rp_x},{rp_y})基站片段数不一致：{frag_counts}"
        print(f"坐标({rp_x},{rp_y})：所有基站滑动后片段数为 {frag_counts[0]}")

        if all_fingerprints:
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
    run(15, '1764748570471')
