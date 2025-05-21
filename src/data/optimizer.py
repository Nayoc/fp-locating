from sklearn.preprocessing import StandardScaler

import torch


class ZScoreNorm:
    """自定义Z-Score归一化工具，支持PyTorch张量和任意维度"""

    def __init__(self, dim=0, eps=1e-8):
        """
        初始化归一化器

        参数:
            dim: 需要归一化的维度。None表示全局归一化，否则按指定维度计算统计量
            eps: 防止除零的小常数
        """
        self.dim = dim
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, x):
        """
        计算并保存归一化所需的均值和标准差

        参数:
            x: 输入的PyTorch张量 [batch_size, ...]
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)

        if self.dim is None:
            # 全局归一化（所有维度）
            self.mean = x.mean()
            self.std = x.std()
        else:
            # 按指定维度计算统计量
            self.mean = x.mean(dim=self.dim, keepdim=True)
            self.std = x.std(dim=self.dim, keepdim=True)

        # 防止标准差为零
        self.std = torch.max(self.std, torch.tensor(self.eps, device=x.device))

        return self

    def norm(self, x):
        """
        应用Z-Score归一化

        参数:
            x: 输入的PyTorch张量

        返回:
            归一化后的张量，保持原始数据类型和维度
        """
        if self.mean is None or self.std is None:
            raise ValueError("请先调用fit方法计算均值和标准差")

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)

        return (x - self.mean) / self.std

    def denorm(self, x_norm):
        """
        反归一化，将数据恢复到原始尺度

        参数:
            x_norm: 归一化后的PyTorch张量

        返回:
            原始尺度的张量
        """
        if self.mean is None or self.std is None:
            raise ValueError("请先调用fit方法计算均值和标准差")

        return x_norm * self.std + self.mean

    def fit_norm(self, x):
        """一次性完成拟合和转换"""
        return self.fit(x).norm(x)



class Norm:
    def __init__(self, norm_params):
        """
        初始化归一化参数
        norm_params: torch.Tensor, shape为(dim_size, 2)，每个维度的 [min_value, max_value]
        """
        self.norm_params = norm_params

    def norm(self, data, dim=1):
        """
        对数据进行归一化
        data: torch.Tensor, 待归一化的数据
        dim: 归一化的维度，默认dim=1
        """
        min_values = self.norm_params[:, 0]
        max_values = self.norm_params[:, 1]

        # 按指定维度进行广播操作以适应data的形状
        shape = [1] * data.ndim  # 初始化形状
        shape[dim] = -1  # 将dim维度的大小设置为-1，以匹配min_values和max_values

        min_values = min_values.view(*shape)
        max_values = max_values.view(*shape)

        # 归一化公式 (data - min) / (max - min)
        normed_data = (data - min_values) / (max_values - min_values)
        return normed_data

    def denorm(self, data, dim=1):
        """
        对数据进行反归一化
        data: torch.Tensor, 已归一化的数据
        dim: 反归一化的维度，默认dim=1
        """
        print(f'data-device:{data.device}')
        print(f'norm_params-device:{self.norm_params.device}')
        if data.device != self.norm_params.device:
            self.norm_params = self.norm_params.to(data.device)
        print(f'data-device:{data.device}')
        print(f'norm_params-device:{self.norm_params.device}')
        min_values = self.norm_params[:, 0]
        max_values = self.norm_params[:, 1]

        # 按指定维度进行广播操作以适应data的形状
        shape = [1] * data.ndim  # 初始化形状
        shape[dim] = -1  # 将dim维度的大小设置为-1，以匹配min_values和max_values

        min_values = min_values.view(*shape)
        max_values = max_values.view(*shape)

        # 反归一化公式 data * (max - min) + min
        denormed_data = data * (max_values - min_values) + min_values
        return denormed_data
