import torch
from torch import nn
from torch.nn import functional as F

import util.coor_utils as cu


class CellCNN(nn.Module):

    def __init__(
            self,
            in_channels: int = 1,
            out_dim: int = 2,
            dropout_rate: float = 0.3,
    ):
        super(CellCNN, self).__init__()
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=10,
            kernel_size=(3, 3),
            stride=1,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(10)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=10,
            out_channels=10,
            kernel_size=(3, 3),
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(10)

        self.conv3 = nn.Conv2d(
            in_channels=10,
            out_channels=5,
            kernel_size=(3, 3),
            stride=1,
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(5)

        self.conv4 = nn.Conv2d(
            in_channels=5,
            out_channels=5,
            kernel_size=(3, 3),
            stride=1,
            padding=1
        )
        self.bn4 = nn.BatchNorm2d(5)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(1, out_dim)

    def _calc_flatten_dim(self, img_h: int, img_w: int) -> int:
        return 5 * img_h * img_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, img_h, img_w = x.shape

        if self.fc.in_features == 1:
            flatten_dim = self._calc_flatten_dim(img_h, img_w)
            self.fc = nn.Linear(flatten_dim, self.fc.out_features).to(x.device)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = x.view(batch_size, -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class MinCNN1D(nn.Module):

    def __init__(self, input_shape, dropout_rate=0.5):
        super(MinCNN1D, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # 1. 创建一个与输入形状匹配的伪数据
        dummy_input = torch.zeros(*input_shape)
        # 2. 计算卷积层输出
        with torch.no_grad():
            conv_output = self.conv_layer(dummy_input)
        # 3. 计算展平后的维度（动态获取全连接层输入尺寸）
        flattened_size = conv_output[0].view(1, -1).size(1)

        # 全连接层：添加Dropout正则化
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        # 第一个卷积块
        x = self.conv_layer(x)

        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc_layers(x)

        return x


# CNN
class MCnn1(nn.Module):
    def __init__(self, input_shape):
        super(MCnn1, self).__init__()

        # 从输入数据的形状中提取通道数和宽度
        _, in_channels, height, width = input_shape

        # 第一层卷积，输入通道根据输入数据调整，输出通道固定为16，卷积核大小(1, 3)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(1, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(16)

        # 第二层卷积
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(32)

        # 第三层卷积
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(64)

        # 计算展平后的尺寸
        self.flattened_size = 64 * height * width

        # 全连接层
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # 第一层卷积
        x = F.relu(self.bn1(self.conv1(x)))

        # 第二层卷积
        x = F.relu(self.bn2(self.conv2(x)))

        # 第三层卷积
        x = F.relu(self.bn3(self.conv3(x)))

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class MCnn2(nn.Module):
    def __init__(self, input_shape):
        super(MCnn2, self).__init__()

        in_channels, height, width = input_shape

        # 调整卷积层配置，更适合(1,6)的输入尺寸
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(1, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))  # 添加池化层减少维度

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(32)

        # 计算展平后的尺寸
        flattened_size = 32 * height * (width // 2)  # 考虑池化后的尺寸变化

        # 全连接层增加神经元数量
        self.fc1 = nn.Linear(flattened_size, 256)
        self.dropout = nn.Dropout(0.5)  # 添加dropout防止过拟合
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)  # 应用池化

        x = F.relu(self.bn2(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 应用dropout
        x = self.fc2(x)

        return x


# LOSS
class CoordinateLoss(nn.Module):
    def __init__(self):
        super(CoordinateLoss, self).__init__()

    def forward(self, predicted_coords, actual_coords):
        # 计算欧几里得距离
        loss = torch.sqrt(torch.sum((predicted_coords - actual_coords) ** 2, dim=1))
        # 计算batch内平均损失
        return torch.mean(loss)


class CombinedLoss(nn.Module):
    def __init__(self, norm, alpha=0.8):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        self.norm = norm

    def forward(self, y_hat, y):
        y_hat = self.norm.denorm(y_hat)
        y = self.norm.denorm(y)
        # 坐标误差（欧几里得距离）
        coord_loss = cu.calc_normal_distance(y_hat, y).mean()
        # 传统MSE损失
        mse_loss = self.mse_loss(y_hat, y)
        # 组合损失
        return self.alpha * coord_loss + (1 - self.alpha) * mse_loss
