import torch


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
