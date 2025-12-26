import torch
from torch import nn
from torch.nn import functional as F

import util.coor_utils as cu


class BasicCnnExtra(nn.Module):
    def __init__(self, in_channels: int = 1, dropout_rate: float = 0.3):
        super().__init__()
        self.dropout_rate = dropout_rate

        #
        self.feature_extractor = nn.Sequential(
            # 卷积块1: in_channels→10
            nn.Conv2d(in_channels, 10, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            # 卷积块2: 10→10
            nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            # 卷积块3: 10→5
            nn.Conv2d(10, 5, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            # 卷积块4: 5→5
            nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(5),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: B x C x H x W
        b, c, h, w = x.size()
        y = x.mean(dim=(2, 3))  # B x C
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# -------------------------
# 融合模块（modal attention + FC 回归头）
# -------------------------
class FusionAttentionRegression(nn.Module):
    def __init__(self, feat_channels: int, fused_hidden: int = 128, cell_bias_init: float = 1.0):
        """
        feat_channels: BasicCnnExtra 最终输出的通道数（例如 5）
        cell_bias_init: 用于让 cell 初始权重偏高（logit bias）
        """
        super().__init__()
        self.feat_channels = feat_channels

        # SE for each branch (we can share or use distinct; 用 distinct)
        self.se_cell = SEBlock(feat_channels, reduction=2)
        self.se_wifi = SEBlock(feat_channels, reduction=2)

        # pool to vector
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 返回 B x C x 1 x 1

        # modality weighting: map each pooled vector -> scalar logit
        self.modality_fc = nn.Sequential(
            nn.Linear(feat_channels, feat_channels // 2),
            nn.ReLU(),
            nn.Linear(feat_channels // 2, 1)
        )

        # we'll keep separate final bias initializations to favor cell
        # but modality_fc is shared, so add learnable scalar biases:
        self.logit_bias_cell = nn.Parameter(torch.tensor(float(cell_bias_init)))
        self.logit_bias_wifi = nn.Parameter(torch.tensor(0.0))

        # final MLP (regression head) from fused vector
        self.regressor = nn.Sequential(
            nn.Linear(feat_channels, fused_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fused_hidden, 2)  # 输出 x,y
        )

    def forward(self, feat_cell, feat_wifi):
        """
        feat_*: B x C x H x W  (C == feat_channels)
        """
        # apply SE
        feat_cell = self.se_cell(feat_cell)
        feat_wifi = self.se_wifi(feat_wifi)

        # pool to vectors
        v_cell = self.pool(feat_cell).view(feat_cell.size(0), -1)  # B x C
        v_wifi = self.pool(feat_wifi).view(feat_wifi.size(0), -1)  # B x C

        # modality logits
        logit_cell = self.modality_fc(v_cell).squeeze(-1) + self.logit_bias_cell  # B
        logit_wifi = self.modality_fc(v_wifi).squeeze(-1) + self.logit_bias_wifi  # B

        logits = torch.stack([logit_cell, logit_wifi], dim=1)  # B x 2
        weights = F.softmax(logits, dim=1)  # B x 2
        # w_cell = weights[:, 0].unsqueeze(-1)  # B x 1
        # w_wifi = weights[:, 1].unsqueeze(-1)
        w_cell = 0.5
        w_wifi = 0.5

        # fused vector
        fused = w_cell * v_cell + w_wifi * v_wifi  # B x C

        # regression head
        out = self.regressor(fused)  # B x 2
        return out


# -------------------------
# 完整封装模型：两个分支（cell: in_channels=3, wifi: in_channels=1）
# -------------------------
class CellWifiFusionModel(nn.Module):
    def __init__(self,
                 cell_in_channels=4,
                 wifi_in_channels=1,
                 base_feat_channels=5,   # 应匹配 BasicCnnExtra 的最终输出通道数 (这里是5)
                 fused_hidden=128,
                 cell_bias_init: float = 1.0):
        super().__init__()
        # 两个分支用 BasicCnnExtra（独立实例）
        self.cell_cnn = BasicCnnExtra(in_channels=cell_in_channels)
        self.wifi_cnn = BasicCnnExtra(in_channels=wifi_in_channels)

        # 确认最后输出通道数（BasicCnnExtra 固定为 5）
        assert base_feat_channels == 5, "Ensure base_feat_channels match BasicCnnExtra's last channel (5)"

        self.fusion = FusionAttentionRegression(feat_channels=base_feat_channels,
                                                fused_hidden=fused_hidden,
                                                cell_bias_init=cell_bias_init)

    def forward(self, x_cell, x_wifi):
        # x_cell: B x 3 x 16 x 3  (例)
        # x_wifi: B x 1 x 16 x 20
        feat_c = self.cell_cnn(x_cell)   # B x C x H x W
        feat_w = self.wifi_cnn(x_wifi)   # B x C x H x W
        out= self.fusion(feat_c, feat_w)
        return out





class BasicCnn(nn.Module):
    def __init__(self, in_channels: int = 1, out_dim: int = 2, dropout_rate: float = 0.3):
        super().__init__()
        self.dropout_rate = dropout_rate

        # 用Sequential封装4个卷积块（Conv→BN→ReLU），精简重复代码
        self.feature_extractor = BasicCnnExtra(in_channels=in_channels, dropout_rate=dropout_rate)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(1, out_dim)  # 占位，后续动态修改

    def _calc_flatten_dim(self, img_h: int, img_w: int) -> int:
        return 5 * img_h * img_w  # 最后一层卷积输出5通道，特征图尺寸=输入尺寸

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, img_h, img_w = x.shape

        # 首次前向时，动态初始化全连接层输入维度
        if self.fc.in_features == 1:
            flatten_dim = self._calc_flatten_dim(img_h, img_w)
            self.fc = nn.Linear(flatten_dim, self.fc.out_features).to(x.device)

        # 特征提取 → Flatten → Dropout → 分类
        x = self.feature_extractor(x)
        x = x.view(batch_size, -1)  # (batch_size, 5*img_h*img_w)
        x = self.dropout(x)
        return self.fc(x)




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
