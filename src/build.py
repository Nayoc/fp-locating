import random
from pathlib import Path

import pandas as pd
import torch

root_dir = str(Path(__file__).parent.parent.parent)
dataset_dir = root_dir + '/data/dataset'

min_rsrp = -140


def syl_translator():
    file = root_dir + "/data/original/syl_data.csv"
    data = pd.read_csv(file)
    x = data['x'].tolist()
    y = data['y'].tolist()

    format_data = []

    ap_list = ['166', '167', '168', '169', '189', '190']

    for index, row in data.iloc[1:].iterrows():

        r = [None] * 8
        for i in range(1, 7):
            apxId = row[str(i) + ' NR PCI']
            apxRsrp = row[str(i) + ' SS RSRP']
            for j in range(len(ap_list)):
                if pd.isnull(apxId):
                    continue

                if str(int(apxId)) == ap_list[j]:
                    r[j] = apxRsrp
        r[6] = x[index]
        r[7] = y[index]
        if pd.isnull(r[0]) and pd.isnull(r[1]) and pd.isnull(r[2]) and pd.isnull(r[3]) and pd.isnull(
                r[4]) and pd.isnull(r[5]):
            continue
        format_data.append(r)

    for i in range(len(format_data)):
        for j in range(len(format_data[i])):
            if pd.isnull(format_data[i][j]):
                if j < 6:
                    format_data[i][j] = -140
                else:
                    del format_data[i]
            if j < 6:
                if format_data[i][j] < -140:
                    format_data[i][j] = -140
                if format_data[i][j] > -30:
                    format_data[i][j] = -30

    new_df = pd.DataFrame(format_data, columns=ap_list + ['x', 'y'])
    new_df.to_csv(root_dir + '/data/format/syl_data.csv', index=False, encoding='utf-8')


def build_dataset(source_file, source_name):
    data = pd.read_csv(root_dir + "/data/format/" + source_file).values.tolist()

    for i in range(len(data)):
        for j in range(len(data[i])):
            if pd.isnull(data[i][j]):
                if j < 6:
                    data[i][j] = min_rsrp
                else:
                    del data[i]
            if j < 6:
                if data[i][j] < -140:
                    data[i][j] = -140
                if data[i][j] > -30:
                    data[i][j] = -30

    valid_data = random_extract(data, 20)
    test_data = random_extract(data, 10)

    train_d, train_l = build_tensor(data)
    valid_d, valid_l = build_tensor(valid_data)
    test_d, test_l = build_tensor(test_data)

    save_cnn_tensor_pth(train_d, train_l, "train", source_name)
    save_cnn_tensor_pth(valid_d, valid_l, "valid", source_name)
    save_cnn_tensor_pth(test_d, test_l, "test", source_name)


def save_cnn_tensor_pth(data, label, type, source: str):
    torch.save({'data': data, 'labels': label},
               dataset_dir + '/cnn/' + source + "/" + type + ".pth")


def build_tensor(data):
    d_tensor = torch.tensor(data)

    d = d_tensor[:, :-2]

    d = torch.unsqueeze(d, dim=1)
    l = d_tensor[:, -2:]

    return d, l


def random_extract(data: [], per: int):
    if per > 100 or per < 0:
        return []

    sample_size = max(1, int(len(data) * (per / 100) + 0.5))  # 向上取整
    # 生成随机索引（不重复）
    indices = random.sample(range(len(data)), sample_size)
    # 提取抽样元素（保留顺序）
    sampled_list = [data[i] for i in indices]

    # 删除原列表中的抽样元素（按索引倒序删除，避免索引错位）
    for i in sorted(indices, reverse=True):
        del data[i]

    return sampled_list


if __name__ == '__main__':
    syl_translator()
    build_dataset("syl_data.csv", "syl")
