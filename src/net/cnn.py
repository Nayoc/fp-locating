# fusion_cnn_attention.py
# 依赖：torch
# 用法：直接运行此文件可做一次 sanity check 前向推理

import torch
import torch.nn as nn
import torch.nn.functional as F

class WifiEncoderCNN(nn.Module):
    """
    Shared 1D-CNN encoder for Wi-Fi per-RP vectors.
    Input: (B, num_rp, num_wifi)  (e.g., B=batch, num_rp=20, num_wifi=20)
    Output: (B, num_rp, out_dim)
    """
    def __init__(self, num_wifi=20, out_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)  # collapse AP dimension
        self.fc = nn.Linear(64, out_dim)
        self.out_dim = out_dim

    def forward(self, wifi):  # wifi: (B, R, W)
        B, R, W = wifi.shape
        x = wifi.view(B * R, 1, W)               # -> (B*R, 1, W)
        x = F.relu(self.bn1(self.conv1(x)))      # -> (B*R, 32, W)
        x = F.relu(self.bn2(self.conv2(x)))      # -> (B*R, 64, W)
        x = self.pool(x).squeeze(-1)             # -> (B*R, 64)
        x = self.fc(x)                           # -> (B*R, out_dim)
        x = x.view(B, R, self.out_dim)           # -> (B, R, out_dim)
        return x

class LteEncoderMLP(nn.Module):
    """
    Small MLP encoder for LTE/NR per-RP scalar(s).
    Input: (B, R, in_dim) (in_dim usually = 1 for RSRP; can extend)
    Output: (B, R, out_dim)
    """
    def __init__(self, in_dim=1, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
            nn.ReLU()
        )
        self.out_dim = out_dim

    def forward(self, lte):  # lte: (B, R, in_dim)
        B, R, D = lte.shape
        x = lte.view(B * R, D)
        x = self.net(x)
        x = x.view(B, R, self.out_dim)
        return x

class PerRPGating(nn.Module):
    """
    Learnable per-RP gate that fuses wifi_emb and lte_emb.
    Gate computed from concatenated features -> sigmoid -> per-channel weights.
    out_emb = gate * wifi_emb + (1-gate) * lte_emb
    """
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()   # produce gating vector in (0,1) per channel
        )

    def forward(self, wifi_emb, lte_emb):
        # wifi_emb, lte_emb: (B, R, D) and D must match
        assert wifi_emb.shape == lte_emb.shape, "Embedding dims must match"
        B, R, D = wifi_emb.shape
        cat = torch.cat([wifi_emb, lte_emb], dim=-1).view(B * R, 2 * D)
        g = self.gate(cat).view(B, R, D)  # (B, R, D)
        out = g * wifi_emb + (1.0 - g) * lte_emb
        return out, g

class RPAttentionPool(nn.Module):
    """
    Attention pooling over RP dimension.
    Input E: (B, R, D). Returns pooled (B, D) and attn weights (B, R).
    """
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, E, mask=None):
        # E: (B, R, D)
        B, R, D = E.shape
        u = torch.tanh(self.proj(E))          # (B, R, D)
        scores = self.v(u).squeeze(-1)        # (B, R)
        if mask is not None:
            # mask: boolean tensor True=valid; we invert for masked_fill
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)      # (B, R)
        attn_exp = attn.unsqueeze(-1)         # (B, R, 1)
        pooled = torch.sum(attn_exp * E, dim=1)  # (B, D)
        return pooled, attn

class FusionCNNAttentionModel(nn.Module):
    """
    Full fusion model:
      - wifi: (B, R, W)
      - lte:  (B, R, L)  (L==1 here, RSRP scalar)
      - rp_mask: (B, R) boolean mask for valid RPs
    Output:
      - coords: (B, 2)
      - aux dict: gate, attn, wifi_emb, lte_emb, fused_rp
    """
    def __init__(self, num_wifi=20, num_rp=20, wifi_dim=64, lte_dim=64, hidden=128, out_dim=2):
        super().__init__()
        self.wifi_enc = WifiEncoderCNN(num_wifi=num_wifi, out_dim=wifi_dim)
        self.lte_enc = LteEncoderMLP(in_dim=1, out_dim=lte_dim)
        # projection if dims differ
        if wifi_dim != lte_dim:
            self.lte_proj = nn.Linear(lte_dim, wifi_dim)
            common_dim = wifi_dim
        else:
            self.lte_proj = None
            common_dim = wifi_dim
        self.gating = PerRPGating(common_dim)
        self.att_pool = RPAttentionPool(common_dim)
        self.head = nn.Sequential(
            nn.Linear(common_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, wifi, lte, rp_mask=None):
        """
        wifi: (B, R, W)
        lte:  (B, R, 1)
        rp_mask: (B, R) bool, True for valid RP
        """
        B, R, W = wifi.shape
        wifi_emb = self.wifi_enc(wifi)      # (B, R, D)
        lte_emb = self.lte_enc(lte)         # (B, R, D_lte)
        if self.lte_proj is not None:
            lte_emb = self.lte_proj(lte_emb)  # -> (B, R, D)
        fused_rp, gate = self.gating(wifi_emb, lte_emb)  # (B, R, D)
        global_vec, attn = self.att_pool(fused_rp, mask=rp_mask)  # (B, D), (B, R)
        coords = self.head(global_vec)  # (B, 2)
        aux = {'gate': gate, 'attn': attn, 'wifi_emb': wifi_emb, 'lte_emb': lte_emb, 'fused_rp': fused_rp}
        return coords, aux

# ---------------------------
# 示例：sanity check 前向推理
# ---------------------------
if __name__ == "__main__":
    B = 4
    R = 20   # RP count
    W = 20   # Wi-Fi per RP
    # 假设数据已归一化到 N(0,1) 或 [-1,1]
    wifi = torch.randn(B, R, W) * 0.5
    lte = torch.randn(B, R, 1) * 0.5
    # 构造 rp_mask (True 表示该 RP 有效)
    rp_mask = torch.ones(B, R, dtype=torch.bool)
    # 模拟第0个样本第5个RP LTE缺失
    rp_mask[0, 5] = False
    lte[0, 5, 0] = -10.0  # sentinel (注意：在实际使用前应做归一化与预处理)

    model = FusionCNNAttentionModel(num_wifi=W, num_rp=R, wifi_dim=64, lte_dim=32, hidden=128)
    coords, aux = model(wifi, lte, rp_mask=rp_mask)

    print("coords:", coords.shape)     # (B, 2)
    print("attn:", aux['attn'].shape)  # (B, R)
    print("gate:", aux['gate'].shape)  # (B, R, D)
