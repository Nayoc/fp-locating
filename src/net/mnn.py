import torch
from torch import nn
from torch.nn import functional as F

import util.coor_utils as cu


class CommonCNN1D(nn.Module):
    """
    多尺度特征1D CNN网络（CommonCNN1D），适用于指纹库定位的坐标回归任务
    处理输入形状：(batch_size, 1, 10) （10个AP的信号数据）
    输出形状：(batch_size, 2) （2维浮点坐标：x, y）
    """

    def __init__(self, dropout_rate=0.3):
        """
        参数:
            dropout_rate: dropout层的丢弃率，防止过拟合
        """
        super(CommonCNN1D, self).__init__()

        # 第一个卷积块：提取细粒度特征（小卷积核）
        self.conv1 = nn.Conv1d(
            in_channels=1,  # 输入通道数（1个通道，对应AP信号）
            out_channels=32,  # 输出通道数（32个特征图）
            kernel_size=3,  # 3个AP的局部特征（细粒度）
            stride=1,  # 步长1
            padding=1  # 保持输出长度与输入一致
        )
        self.bn1 = nn.BatchNorm1d(32)  # 批量归一化，加速训练
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 下采样

        # 第二个卷积块：提取中尺度特征
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,  # 5个AP的局部特征（中尺度）
            stride=1,
            padding=2
        )
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 第三个卷积块：提取全局特征（大卷积核）
        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=7,  # 7个AP的局部特征（全局关联）
            stride=1,
            padding=3
        )
        self.bn3 = nn.BatchNorm1d(128)

        # 全连接层：回归出2维坐标
        self.fc1 = nn.Linear(128 * 1, 256)  # 输入维度根据卷积输出计算
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)  # 输出2个值：x坐标和y坐标

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重，提升训练效果"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播过程
        参数:
            x: 输入张量，形状为 (batch_size, 1, 10)
        返回:
            输出张量，形状为 (batch_size, 2)，包含x和y坐标
        """
        # 第一个卷积块
        x = self.conv1(x)  # 输出: (batch_size, 32, 10)
        x = self.bn1(x)  # 输出: (batch_size, 32, 10)
        x = F.relu(x)  # 输出: (batch_size, 32, 10)
        x = self.pool1(x)  # 输出: (batch_size, 32, 5) （10/2=5）

        # 第二个卷积块
        x = self.conv2(x)  # 输出: (batch_size, 64, 5)
        x = self.bn2(x)  # 输出: (batch_size, 64, 5)
        x = F.relu(x)  # 输出: (batch_size, 64, 5)
        x = self.pool2(x)  # 输出: (batch_size, 64, 2) （5//2=2）

        # 第三个卷积块
        x = self.conv3(x)  # 输出: (batch_size, 128, 2)
        x = self.bn3(x)  # 输出: (batch_size, 128, 2)
        x = F.relu(x)  # 输出: (batch_size, 128, 2)

        # 全局平均池化：减少参数，增强泛化
        x = F.adaptive_avg_pool1d(x, 1)  # 输出: (batch_size, 128, 1)
        x = x.view(x.size(0), -1)  # 展平: (batch_size, 128*1)

        # 全连接层
        x = self.fc1(x)  # 输出: (batch_size, 256)
        x = self.bn_fc1(x)  # 输出: (batch_size, 256)
        x = F.relu(x)  # 输出: (batch_size, 256)
        x = self.dropout(x)  # 输出: (batch_size, 256)
        x = self.fc2(x)  # 输出: (batch_size, 128)
        x = F.relu(x)  # 输出: (batch_size, 128)
        x = self.fc3(x)  # 输出: (batch_size, 2) （最终2维坐标）

        return x


class MinCNN1D(nn.Module):

    def __init__(self,input_shape, dropout_rate=0.5):

        super(MinCNN1D, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
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
