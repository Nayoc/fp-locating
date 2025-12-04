import os
from pathlib import Path

import torch

import exec.mtrain as train
import net.mnn as mnn
from util.txt_utils import TxtArrayTool

root_dir = str(Path(os.path.abspath(__file__)).parent.parent)
data_dir = root_dir + '/data'


def run(dataset_url, ap_list, model_file):
    text_tool = TxtArrayTool()
    header_path = data_dir + dataset_url
    wifi_order = text_tool.read(header_path + '/wifi_order.txt')
    cell_order = text_tool.read(header_path + '/cell_order.txt')

    cell_data = []
    wifi_data = []
    for rp in ap_list:
        if rp['source'] == 'cell':
            cell_data.append(rp)
        elif rp['source'] == 'wifi':
            wifi_data.append(rp)

    cell_set = []
    wifi_set = []
    # 1. 处理Cell数据：n×16×3（n=Cell AP数量）
    cell_ap_dict = {}
    for rp in cell_data:
        ap_id = rp['apId']
        rsrp = rp.get('rsrp', -120)
        rsrq = rp.get('rsrq', -20)
        sinr = rp.get('sinr', 0)
        cell_ap_dict[ap_id] = [rsrp, rsrq, sinr]

    n_cell_channels = len(cell_order)
    if n_cell_channels == 0:
        cell_feat = torch.full((3, 16, 3), fill_value=-120)
        cell_feat[:, :, 1] = -20
        cell_feat[:, :, 2] = -10
    else:
        cell_channel_list = []
        for ap_id in cell_order:
            if ap_id in cell_ap_dict:
                channel = torch.tensor(cell_ap_dict[ap_id], dtype=torch.float32).unsqueeze(0).repeat(16, 1)
                cell_channel_list.append(channel)
            else:
                channel = torch.full((16, 3), fill_value=-120)
                channel[:, 1] = -20
                channel[:, 2] = -10
                cell_channel_list.append(channel)
        cell_feat = torch.stack(cell_channel_list, dim=0)
    cell_set.append(cell_feat)

    # 2. 处理WiFi数据：1×16×10（按header顺序取前10个AP）
    wifi_rssi_list = []
    for ap_id in wifi_order[:10]:
        rssi = -100
        for rp in wifi_data:
            if rp['apId'] == ap_id:
                rssi = rp.get('rssi', -100)
                break
        wifi_rssi_list.append(rssi)
    wifi_feat = torch.tensor(wifi_rssi_list, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    wifi_feat = wifi_feat.repeat(1, 16, 1)
    wifi_set.append(wifi_feat)

    # 3. 转换为tensor（适配batch维度）
    cell_set = torch.stack(cell_set, dim=0)
    wifi_set = torch.stack(wifi_set, dim=0)
    # -------------------------- 插入逻辑结束 --------------------------

    print(f"Cell输入shape: {cell_set.shape}")  # 打印验证：(batch, n, 16, 3)
    print(f"WiFi输入shape: {wifi_set.shape}")  # 打印验证：(batch, 1, 16, 10)

    net = mnn.CellWifiFusionModel(cell_in_channels=cell_set.shape[1], wifi_in_channels=1,
                                  base_feat_channels=5,
                                  fused_hidden=128, cell_bias_init=1.0)

    try:
        net = train.load_model(net, model_file, mode='eval')

        data = {'cell': cell_set, 'wifi': wifi_set}
        predications = train.calculate(net, train.try_gpu(), data, model=0)

        pred_cpu = predications.cpu()  # 转移到CPU（避免GPU tensor无法直接转标量）
        pred_np = pred_cpu.squeeze(0).numpy()
        x = round(pred_np[0].item(), 2)
        y = round(pred_np[1].item(), 2)  # 转为Python float
        print('predications——>x:' + str(x) + ",y:" + str(y))
        return x, y
    except Exception as e:
        raise



if __name__ == '__main__':
    run([])
