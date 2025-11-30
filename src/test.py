import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

# ===================== 1. 自定义注意力融合层（核心：含可学习权重的全连接层） =====================
class BaseWiFiAttentionFusion(nn.Module):
    def __init__(self):
        super(BaseWiFiAttentionFusion, self).__init__()
        self.weight_linear = nn.Linear(48, 2)  # 可学习参数：weight(2×48) + bias(2×1)
        self.sigmoid = nn.Sigmoid()  # 激活函数：将输出限制在0~1，符合权重范围
        # WiFi升维层（16通道→32通道，与基站通道一致）
        self.wifi_upconv = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1, padding='same')

    def forward(self, base_feat, wifi_feat):
        """
        前向传播：注意力加权融合
        :param base_feat: 基站分支特征，shape=(batch, 32, 16) （batch×通道数×时序窗口）
        :param wifi_feat: WiFi分支特征，shape=(batch, 16, 16)
        :return: 融合后特征，shape=(batch, 32, 16)
        """
        # 步骤1：全局平均池化（GAP）→ 压缩时序维度（16→1），提取全局特征（PyTorch实现）
        # dim=2：对时序维度（第3维）求平均，保留batch和通道维度
        base_gap = torch.mean(base_feat, dim=2)  # (batch, 32, 16) → (batch, 32)
        wifi_gap = torch.mean(wifi_feat, dim=2)  # (batch, 16, 16) → (batch, 16)

        # 步骤2：拼接全局特征，通过可学习全连接层编码权重候选值（机器学习核心步骤）
        concat_gap = torch.cat([base_gap, wifi_gap], dim=1)  # (batch, 32+16) = (batch, 48)
        init_weights = self.weight_linear(concat_gap)  # (batch, 48) → (batch, 2)：随机初始化→逐步学习
        init_weights = self.sigmoid(init_weights)  # 激活后：(batch, 2) ∈ [0,1]

        # 步骤3：提取权重，强制基站权重≥0.6（主辅规则约束）
        base_w = init_weights[:, 0:1]  # (batch, 1)：基站初始权重
        wifi_w = init_weights[:, 1:2]  # (batch, 1)：WiFi初始权重
        # 基站权重放大后裁剪：确保≥0.6，≤1.0
        base_w = torch.clamp(base_w * 1.2, min=0.6, max=1.0)
        # WiFi权重=1-基站权重：自动≤0.4
        wifi_w = 1 - base_w

        # 步骤4：权重广播（PyTorch实现：扩展维度适配特征图）
        # 基站权重：(batch,1) → (batch, 32, 16)：每个通道×每个时序窗口都用同一个基站权重
        base_w_broadcast = base_w.unsqueeze(1).unsqueeze(2).expand_as(base_feat)
        # WiFi权重：(batch,1) → (batch, 16, 16)：适配WiFi分支特征维度
        wifi_w_broadcast = wifi_w.unsqueeze(1).unsqueeze(2).expand_as(wifi_feat)

        # 步骤5：WiFi特征升维，与基站特征维度一致
        wifi_feat_up = self.wifi_upconv(wifi_feat)  # (batch,16,16) → (batch,32,16)

        # 步骤6：加权融合（逐元素相乘+求和）
        fused_feat = base_feat * base_w_broadcast + wifi_feat_up * wifi_w_broadcast

        # 可选：记录权重（训练时查看）
        self.base_weight = torch.mean(base_w).item()
        self.wifi_weight = torch.mean(wifi_w).item()

        return fused_feat

# ===================== 2. 完整融合模型（基站主分支+WiFi辅分支+注意力融合） =====================
class BaseWiFiFusionModel(nn.Module):
    def __init__(self, n_reference_points=41):
        super(BaseWiFiFusionModel, self).__init__()
        self.n_reference = n_reference_points

        # ---------------------- 基站主分支（提取稳定特征） ----------------------
        self.base_branch = nn.Sequential(
            # Conv1d：in_channels=3（基站3种信号），out_channels=32（特征通道），kernel_size=3
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, padding='same'),
            nn.BatchNorm1d(num_features=32),  # 1D BatchNorm：num_features=卷积输出通道数
            nn.ReLU(inplace=True)  # 激活函数：增强非线性，无池化（保留时序维度）
        )

        # ---------------------- WiFi辅分支（提取补充特征） ----------------------
        self.wifi_branch = nn.Sequential(
            # Conv1d：in_channels=20（WiFi 20个AP/特征），out_channels=16（轻量化设计）
            nn.Conv1d(in_channels=20, out_channels=16, kernel_size=3, padding='same'),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(inplace=True)
        )

        # ---------------------- 注意力融合层 ----------------------
        self.attention_fusion = BaseWiFiAttentionFusion()

        # ---------------------- 全局特征整合+分类输出 ----------------------
        self.classifier = nn.Sequential(
            # 全局平均池化：压缩时序维度（16→1），输出(batch,32)
            nn.AdaptiveAvgPool1d(1),  # 等价于torch.mean(dim=2)，更灵活
            nn.Flatten(),  # 展平：(batch,32,1) → (batch,32)
            nn.Linear(32, 64),  # 全连接层：特征编码
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),  # 防过拟合（小数据集必备）
            nn.Linear(64, self.n_reference)  # 输出层：对应参考点数量
        )

    def forward(self, base_input, wifi_input):
        """
        前向传播：双输入→双分支→融合→分类
        :param base_input: 基站输入，shape=(batch, 3, 16)
        :param wifi_input: WiFi输入，shape=(batch, 20, 16)
        :return: 参考点分类logits（未经过softmax，PyTorch损失函数会自动处理）
        """
        # 1. 分支特征提取
        base_feat = self.base_branch(base_input)  # (batch,3,16) → (batch,32,16)
        wifi_feat = self.wifi_branch(wifi_input)  # (batch,20,16) → (batch,16,16)

        # 2. 注意力融合
        fused_feat = self.attention_fusion(base_feat, wifi_feat)  # (batch,32,16)

        # 3. 分类输出
        logits = self.classifier(fused_feat)  # (batch, n_reference)

        return logits

# ===================== 3. 数据集类（适配PyTorch DataLoader） =====================
class FusionDataset(Dataset):
    def __init__(self, base_data, wifi_data, labels):
        """
        :param base_data: 基站数据，shape=(n_samples, 3, 16)
        :param wifi_data: WiFi数据，shape=(n_samples, 20, 16)
        :param labels: 参考点ID标签，shape=(n_samples,)
        """
        self.base_data = torch.tensor(base_data, dtype=torch.float32)
        self.wifi_data = torch.tensor(wifi_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)  # 分类任务标签用long类型

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.base_data[idx], self.wifi_data[idx], self.labels[idx]

# ===================== 4. 训练与预测示例 =====================
if __name__ == "__main__":
    # ---------------------- 配置参数 ----------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动检测GPU/CPU
    batch_size = 16
    n_samples = 1000  # 小数据集示例
    n_reference = 41  # 参考点数量
    epochs = 50
    learning_rate = 5e-5

    # ---------------------- 1. 模拟训练数据（适配输入shape） ----------------------
    # 基站数据：(n_samples, 3, 16) → 模拟RSRP/SINR（-120~-60）
    base_data = np.random.uniform(low=-120, high=-60, size=(n_samples, 3, 16))
    # WiFi数据：(n_samples, 20, 16) → 模拟20个AP的RSSI（-100~-30）
    wifi_data = np.random.uniform(low=-100, high=-30, size=(n_samples, 20, 16))
    # 标签：(n_samples,) → 0~n_reference-1的整数
    labels = np.random.randint(low=0, high=n_reference, size=(n_samples,))

    # ---------------------- 2. 数据预处理：归一化（提升训练稳定性） ----------------------
    # 基站数据归一化：(-120~-60) → (0~1)
    base_data = (base_data - (-120)) / ((-60) - (-120))  # (x - min) / (max - min)
    # WiFi数据归一化：(-100~-30) → (0~1)
    wifi_data = (wifi_data - (-100)) / ((-30) - (-100))

    # ---------------------- 3. 构建数据集和DataLoader ----------------------
    dataset = FusionDataset(base_data, wifi_data, labels)
    # 划分训练集（80%）和验证集（20%）
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ---------------------- 4. 初始化模型、损失函数、优化器 ----------------------
    model = BaseWiFiFusionModel(n_reference_points=n_reference).to(device)  # 模型移到GPU/CPU
    criterion = nn.CrossEntropyLoss()  # 分类任务损失函数（自动处理logits和long标签）
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 优化器（更新所有可学习参数）

    # ---------------------- 5. 训练循环（机器学习核心流程：参数更新） ----------------------
    model.train()  # 模型设为训练模式（BatchNorm、Dropout生效）
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (base_batch, wifi_batch, label_batch) in enumerate(train_loader):
            # 数据移到GPU/CPU
            base_batch = base_batch.to(device)
            wifi_batch = wifi_batch.to(device)
            label_batch = label_batch.to(device)

            # 步骤1：梯度清零（PyTorch需手动清零，避免梯度累积）
            optimizer.zero_grad()

            # 步骤2：前向传播（计算预测logits）
            logits = model(base_batch, wifi_batch)  # (batch, n_reference)

            # 步骤3：计算损失（反馈信号：预测与真实标签的差距）
            loss = criterion(logits, label_batch)

            # 步骤4：反向传播（自动计算所有可学习参数的梯度）
            loss.backward()  # 核心：PyTorch autograd引擎计算梯度

            # 步骤5：参数更新（优化器按梯度调整参数）
            optimizer.step()  # 核心：更新self.weight_linear等层的weight和bias

            # 统计训练指标
            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)  # 取概率最高的参考点ID
            total += label_batch.size(0)
            correct += (predicted == label_batch).sum().item()

        # 计算epoch指标
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        # 验证集评估（省略，可参考训练流程，用model.eval()和torch.no_grad()）

        # 打印训练信息（含注意力权重）
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Base Attention Weight: {model.attention_fusion.base_weight:.2f}, WiFi Weight: {model.attention_fusion.wifi_weight:.2f}')

    # ---------------------- 6. 预测示例 ----------------------
    model.eval()  # 模型设为评估模式（BatchNorm、Dropout失效）
    with torch.no_grad():  # 禁用梯度计算，提升速度
        # 模拟单条测试数据
        test_base = np.random.uniform(low=-120, high=-60, size=(1, 3, 16))
        test_wifi = np.random.uniform(low=-100, high=-30, size=(1, 20, 16))
        # 归一化
        test_base = (test_base - (-120)) / 60
        test_wifi = (test_wifi - (-100)) / 70
        # 转Tensor并移到设备
        test_base = torch.tensor(test_base, dtype=torch.float32).to(device)
        test_wifi = torch.tensor(test_wifi, dtype=torch.float32).to(device)

        # 预测
        logits = model(test_base, test_wifi)
        pred_prob = F.softmax(logits, dim=1)  # 转换为概率分布
        pred_reference_id = torch.argmax(pred_prob, dim=1).item()

        # 模拟参考点坐标库
        reference_coords = {i: (i*1.0, i*0.8) for i in range(n_reference)}
        final_location = reference_coords[pred_reference_id]

        print(f'\n预测参考点ID：{pred_reference_id}')
        print(f'最终定位坐标：{final_location}')
        print(f'注意力权重（基站/WiFi）：{model.attention_fusion.base_weight:.2f}/{model.attention_fusion.wifi_weight:.2f}')