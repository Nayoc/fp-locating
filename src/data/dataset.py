import torch
from torch.utils.data import Dataset

class FusionDataset(Dataset):
    def __init__(self, base_data, wifi_data, labels):
        """
        :param base_data: 基站数据，shape=(n_samples, 3, 16)
        :param wifi_data: WiFi数据，shape=(n_samples, 20, 16)
        :param labels: 参考点坐标标签，shape=(n_samples, 2)
        """
        self.base_data = torch.tensor(base_data, dtype=torch.float32)
        self.wifi_data = torch.tensor(wifi_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.base_data[idx], self.wifi_data[idx], self.labels[idx]
