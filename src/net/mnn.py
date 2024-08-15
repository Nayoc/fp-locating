from torch import nn
import torch
import torchvision.models as models
from torch.nn import functional as F
import src.util.coor_utils as cu
import src.data.optimizer as optimizer


# 深度学习模型
class MCNN1(nn.Module):
    def __init__(self, input_shape):
        super(MCNN1, self).__init__()

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


class CoordinateLoss(nn.Module):
    def __init__(self):
        super(CoordinateLoss, self).__init__()

    def forward(self, predicted_coords, actual_coords):
        # 计算欧几里得距离
        loss = torch.sqrt(torch.sum((predicted_coords - actual_coords) ** 2, dim=1))
        # 计算batch内平均损失
        return torch.mean(loss)


class CombinedLoss(nn.Module):
    def __init__(self, norm, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        self.norm = norm

    def forward(self, y_hat, y):
        # 传统MSE损失
        mse_loss = self.mse_loss(y_hat, y)

        y_hat = self.norm.denorm(y_hat)
        y = self.norm.denorm(y)
        # 坐标误差（欧几里得距离）
        coord_loss = cu.calc_geodesic_distance(y_hat, y).mean()
        # 组合损失
        return self.alpha * coord_loss + (1 - self.alpha) * mse_loss


# 使用自定义的坐标误差损失
loss_fn = CoordinateLoss()
