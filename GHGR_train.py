import warnings

warnings.filterwarnings("ignore")

# 修复OpenBLAS线程问题（保持不变）
import os

os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

# 显存分配优化（保持不变）
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:False"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn import functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import gc

# 设置matplotlib支持中文（保持不变）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# --------------------------
# GPU配置（保持不变）
# --------------------------
def setup_gpu():
    """配置GPU设置（降低使用率，预留更多资源）"""
    print("🔧 配置GPU设置（降低使用率模式）...")

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ 找到 {gpu_count} 个GPU")

        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

        device = torch.device("cuda:0")
        torch.cuda.set_per_process_memory_fraction(0.7, device=device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)} (限制使用70%显存，禁用TF32加速)")
        return device, True
    else:
        print("❌ 未找到GPU，将使用CPU进行训练")
        return torch.device("cpu"), False


device, GPU_AVAILABLE = setup_gpu()


# --------------------------
# 核心优化：改进中心损失模块
# --------------------------
class ImprovedCenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device, alpha=0.9):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.alpha = alpha  # 中心更新动量系数
        self.initialized = False  # 标记是否已初始化

    def initialize_centers(self, feat, labels):
        """使用训练数据的类别均值初始化中心"""
        if self.initialized:
            return

        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                class_feat = feat[mask]
                self.centers.data[c] = class_feat.mean(dim=0).detach()

        self.initialized = True
        print(f"✅ 中心损失已使用训练数据类别均值初始化")

    def forward(self, feat, labels, use_momentum=True):
        # 特征L2归一化
        feat = F.normalize(feat, p=2, dim=1)
        centers = F.normalize(self.centers, p=2, dim=1)

        batch_size = feat.size(0)

        # 计算距离矩阵（数值稳定版）
        distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(feat, centers.t(), beta=1, alpha=-2)
        distmat = distmat.clamp(min=1e-6)  # 避免负距离

        # 计算类别掩码（仅用于损失计算，不影响后续动量更新）
        classes = torch.arange(self.num_classes).long().to(self.device)
        labels_expand = labels.unsqueeze(1).expand(batch_size, self.num_classes)  # 重命名为labels_expand，区分原始labels
        mask = labels_expand.eq(classes.expand(batch_size, self.num_classes))

        # 计算类别内距离
        dist = distmat * mask.float()
        center_loss = dist.sum() / batch_size

        # 动量更新中心（使用原始一维labels）
        if self.training and use_momentum and self.initialized:
            for c in range(self.num_classes):
                mask_c = (labels == c)  # 使用原始一维labels生成掩码
                if mask_c.sum() > 0:
                    feat_c = feat[mask_c].mean(dim=0)
                    # 动量更新：新中心 = 动量*旧中心 + (1-动量)*当前类别均值
                    self.centers.data[c] = self.alpha * self.centers.data[c] + (1 - self.alpha) * feat_c

        return center_loss


# --------------------------
# 其他模型模块保持不变
# --------------------------
class StatsSelfAttention(nn.Module):
    """统计特征自注意力模块（自适应选择有效特征）"""

    def __init__(self, input_dim, embed_dim=64, num_heads=4, dropout=0.2):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"

        self.proj = nn.Linear(input_dim, embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x_proj = self.proj(x)
        x_seq = x_proj.unsqueeze(1)

        attn_out, attn_weights = self.self_attn(
            query=x_seq,
            key=x_seq,
            value=x_seq
        )
        x_attn = self.layer_norm(x_seq + self.dropout(attn_out))
        output = self.output_proj(x_attn).squeeze(1)

        return output, attn_weights


class TemporalBlock(nn.Module):
    """TCN时序块（扩张卷积）- 确保输入输出seq_len和feature_dim都匹配"""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super().__init__()
        padding = (dilation * (kernel_size - 1)) // 2
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.GELU()
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(out + residual)


class TemporalConvNet(nn.Module):
    """时序卷积网络（TCN）- 输出feature_dim与GRU一致（128维）"""

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                     stride=1, dilation=dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class GRU_TCN_Module(nn.Module):
    """GRU+TCN混合时序提取模块（确保两者输出维度一致）"""

    def __init__(self, input_dim, gru_hidden=64, tcn_channels_list=[64, 64, 128], kernel_size=3):
        super().__init__()
        self.gru_hidden = gru_hidden
        self.fusion_dim = gru_hidden * 2

        self.bigru = nn.GRU(input_dim, gru_hidden, 2, batch_first=True, bidirectional=True, dropout=0.2)
        self.tcn = TemporalConvNet(input_dim, tcn_channels_list, kernel_size, dropout=0.2)

        self.gate = nn.Sequential(
            nn.Linear(self.fusion_dim + tcn_channels_list[-1], 1),
            nn.Sigmoid()
        )

        self.feature_proj = nn.Linear(self.fusion_dim, self.fusion_dim)

    def forward(self, x):
        gru_out, _ = self.bigru(x)
        tcn_out = self.tcn(x.transpose(1, 2)).transpose(1, 2)

        combined = torch.cat([gru_out, tcn_out], dim=-1)
        gate_weight = self.gate(combined)
        fused_features = gate_weight * gru_out + (1 - gate_weight) * tcn_out

        output = self.feature_proj(fused_features)
        return output


class CrossModalMutualAttention(nn.Module):
    """统计-序列跨模态互注意力融合（确保输入维度匹配）"""

    def __init__(self, stats_dim=128, seq_feature_dim=128, fusion_dim=128):
        super().__init__()
        self.stats_proj = nn.Linear(stats_dim, fusion_dim)
        self.seq_proj = nn.Linear(seq_feature_dim, fusion_dim)

        self.num_heads = 4
        assert fusion_dim % self.num_heads == 0
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=self.num_heads,
            dropout=0.2,
            batch_first=True
        )

        self.stats_enhance = nn.Linear(fusion_dim * 2, fusion_dim)
        self.seq_enhance = nn.Linear(fusion_dim * 2, fusion_dim)
        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2)
        )

    def forward(self, stats_feat, seq_feat):
        stats_proj = self.stats_proj(stats_feat).unsqueeze(1)
        seq_proj = self.seq_proj(seq_feat)

        seq_attended, _ = self.attention(query=stats_proj, key=seq_proj, value=seq_proj)
        stats_attended, _ = self.attention(query=seq_proj, key=stats_proj, value=stats_proj)

        stats_enhanced = self.stats_enhance(torch.cat([stats_proj.squeeze(1), seq_attended.squeeze(1)], dim=-1))
        seq_enhanced = self.seq_enhance(torch.cat([seq_proj, stats_attended], dim=-1))

        seq_pooled = seq_enhanced.mean(dim=1)
        fused_feat = self.final_fusion(torch.cat([stats_enhanced, seq_pooled], dim=-1))

        return fused_feat


class MultiScaleConvBlock(nn.Module):
    """多尺度卷积块"""

    def __init__(self, in_channels, out_channels=64, kernel_sizes=[3, 5, 7], dropout_rate=0.2):
        super(MultiScaleConvBlock, self).__init__()

        self.conv_branches = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = (kernel_size - 1) // 2
            conv_branch = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
            self.conv_branches.append(conv_branch)

        self.output_fusion = nn.Conv1d(out_channels * len(kernel_sizes), out_channels, 1)

    def forward(self, x):
        branch_outputs = []
        for conv_branch in self.conv_branches:
            branch_outputs.append(conv_branch(x))

        concatenated = torch.cat(branch_outputs, dim=1)
        fused_output = self.output_fusion(concatenated)

        return fused_output


class ChannelSELayer(nn.Module):
    """通道注意力机制"""

    def __init__(self, channel, reduction=16):
        super(ChannelSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(channel // reduction, 4)),
            nn.GELU(),
            nn.Linear(max(channel // reduction, 4), channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class EnhancedResidualBlock(nn.Module):
    """增强残差块"""

    def __init__(self, input_dim, hidden_dim=None, dropout_rate=0.2):
        super(EnhancedResidualBlock, self).__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        self.activation = nn.GELU()

        self.residual_adjust = None
        if input_dim != hidden_dim:
            self.residual_adjust = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        residual = x

        out = self.network(x)

        if self.residual_adjust is not None:
            residual = self.residual_adjust(residual)
        elif residual.shape != out.shape:
            diff = out.shape[1] - residual.shape[1]
            residual = F.pad(residual, (0, diff))

        out += residual
        return self.activation(out)


class GRUFirstTrafficClassifier(nn.Module):
    """先GRU-TCN混合时序提取 + 跨模态互注意力融合的流量分类模型"""

    def __init__(self, stats_dim, seq_length, seq_channels, num_classes=3, dropout_rate=0.25):
        super(GRUFirstTrafficClassifier, self).__init__()

        print(
            f"GRU-TCN混合模型初始化: stats_dim={stats_dim}, seq_length={seq_length}, seq_channels={seq_channels}, num_classes={num_classes}")

        self.stats_self_attn = StatsSelfAttention(
            input_dim=stats_dim,
            embed_dim=64,
            num_heads=4,
            dropout=dropout_rate
        )

        self.stats_branch = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            EnhancedResidualBlock(64, 128, dropout_rate),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.stats_output_dim = 128

        self.stats_classifier = nn.Sequential(
            nn.Linear(self.stats_output_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

        self.gru_tcn = GRU_TCN_Module(
            input_dim=seq_channels,
            gru_hidden=64,
            tcn_channels_list=[64, 64, 128],
            kernel_size=3
        )
        self.gru_tcn_output_dim = 128

        self.multiscale_conv1 = MultiScaleConvBlock(
            in_channels=self.gru_tcn_output_dim,
            out_channels=64,
            kernel_sizes=[3, 5, 7],
            dropout_rate=dropout_rate
        )

        self.se_attention1 = ChannelSELayer(64)

        self.multiscale_conv2 = MultiScaleConvBlock(
            64,
            out_channels=32,
            kernel_sizes=[3, 5, 7],
            dropout_rate=dropout_rate
        )

        self.se_attention2 = ChannelSELayer(32)

        self.seq_processor = nn.Sequential(
            nn.Linear(32, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.GELU()
        )
        self.seq_output_dim = 128

        self.seq_classifier = nn.Sequential(
            nn.Linear(self.seq_output_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

        self.feature_fusion = CrossModalMutualAttention(
            stats_dim=self.stats_output_dim,
            seq_feature_dim=self.gru_tcn_output_dim,
            fusion_dim=128
        )
        self.fusion_output_dim = 64

        self.classifier = nn.Sequential(
            EnhancedResidualBlock(self.fusion_output_dim, 64, 0.3),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, stats, seq, return_branches=False, return_attn_weights=False):
        stats_attn, attn_weights = self.stats_self_attn(stats)
        stats_feat = self.stats_branch(stats_attn)
        stats_output = self.stats_classifier(stats_feat)

        gru_tcn_out = self.gru_tcn(seq)
        gru_tcn_out_transposed = gru_tcn_out.transpose(1, 2)
        conv1_out = self.multiscale_conv1(gru_tcn_out_transposed)
        attn1_out = self.se_attention1(conv1_out)
        conv2_out = self.multiscale_conv2(attn1_out)
        attn2_out = self.se_attention2(conv2_out)
        avg_pool = torch.mean(attn2_out, dim=2)
        seq_feat = self.seq_processor(avg_pool)
        seq_output = self.seq_classifier(seq_feat)

        fused_feat = self.feature_fusion(stats_feat, gru_tcn_out)
        fused_output = self.classifier(fused_feat)

        if return_branches and return_attn_weights:
            return stats_output, seq_output, fused_output, attn_weights, stats_feat
        elif return_branches:
            return stats_output, seq_output, fused_output
        elif return_attn_weights:
            return fused_output, attn_weights
        else:
            return fused_output


def create_gru_first_model(stats_shape, seq_shape, num_classes=3):
    """创建整合创新模块的PyTorch模型（保持原始复杂度）"""
    if len(stats_shape) != 1:
        raise ValueError(f"统计特征形状错误：预期1维，实际{len(stats_shape)}维")
    if len(seq_shape) != 2:
        raise ValueError(f"序列特征形状错误：预期2维，实际{len(seq_shape)}维")

    stats_dim = stats_shape[0]
    seq_length, seq_channels = seq_shape

    print(
        f"创建GRU-TCN混合模型: stats_dim={stats_dim}, seq_length={seq_length}, seq_channels={seq_channels}, num_classes={num_classes}")
    print("🔍 统计特征分支新增：自注意力机制（特征自适应选择）+ 改进版中心损失（解决停滞问题）")
    print("🔍 新增分支损失优化：统计特征分类损失 + 序列特征分类损失")

    model = GRUFirstTrafficClassifier(
        stats_dim=stats_dim,
        seq_length=seq_length,
        seq_channels=seq_channels,
        num_classes=num_classes
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GRU-TCN混合模型参数量：总参数={total_params:,}，可训练参数={trainable_params:,}（原始复杂度不变）")

    return model.to(device)


# --------------------------
# 训练相关函数（核心优化）
# --------------------------
class TestAccuracyCallback:
    """测试集准确率回调（增强分支准确率监控）"""

    def __init__(self, model_save_path, X_stats_test, X_seqs_test, y_test,
                 save_best_only=True, patience=10, eval_batch_size=64):
        self.model_save_path = model_save_path
        self.X_stats_test = X_stats_test
        self.X_seqs_test = X_seqs_test
        self.y_test = y_test
        self.save_best_only = save_best_only
        self.patience = patience
        self.eval_batch_size = eval_batch_size
        self.best_accuracy = 0.0
        self.best_stats_accuracy = 0.0
        self.best_seq_accuracy = 0.0
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, model, optimizer, center_optimizer=None):
        model.eval()
        total_correct_stats = 0
        total_correct_seq = 0
        total_correct_fused = 0
        total_samples = 0
        total_loss_stats = 0.0
        total_loss_seq = 0.0
        total_loss_fused = 0.0
        n_batches = (len(self.X_stats_test) + self.eval_batch_size - 1) // self.eval_batch_size

        criterion_cls = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.eval_batch_size
                end_idx = min((batch_idx + 1) * self.eval_batch_size, len(self.X_stats_test))

                stats_batch = torch.FloatTensor(self.X_stats_test[start_idx:end_idx]).to(device)
                seqs_batch = torch.FloatTensor(self.X_seqs_test[start_idx:end_idx]).to(device)
                y_batch = torch.FloatTensor(self.y_test[start_idx:end_idx]).to(device)
                y_batch_int = torch.argmax(y_batch, dim=1)

                stats_output, seq_output, fused_output = model(stats_batch, seqs_batch, return_branches=True)

                # 计算各分支损失
                loss_stats = criterion_cls(stats_output, y_batch_int)
                loss_seq = criterion_cls(seq_output, y_batch_int)
                loss_fused = criterion_cls(fused_output, y_batch_int)

                total_loss_stats += loss_stats.item() * len(y_batch)
                total_loss_seq += loss_seq.item() * len(y_batch)
                total_loss_fused += loss_fused.item() * len(y_batch)

                predicted_stats = torch.argmax(stats_output, dim=1)
                total_correct_stats += (predicted_stats == y_batch_int).sum().item()

                predicted_seq = torch.argmax(seq_output, dim=1)
                total_correct_seq += (predicted_seq == y_batch_int).sum().item()

                predicted_fused = torch.argmax(fused_output, dim=1)
                total_correct_fused += (predicted_fused == y_batch_int).sum().item()

                total_samples += len(y_batch)

        test_accuracy_stats = total_correct_stats / total_samples if total_samples > 0 else 0.0
        test_accuracy_seq = total_correct_seq / total_samples if total_samples > 0 else 0.0
        test_accuracy_fused = total_correct_fused / total_samples if total_samples > 0 else 0.0
        test_loss_stats = total_loss_stats / total_samples if total_samples > 0 else 0.0
        test_loss_seq = total_loss_seq / total_samples if total_samples > 0 else 0.0
        test_loss_fused = total_loss_fused / total_samples if total_samples > 0 else 0.0

        print(f"\nEpoch {epoch + 1} 测试集评估:")
        print(f"  统计特征分支: 准确率={test_accuracy_stats:.4f}, 损失={test_loss_stats:.4f}")
        print(f"  序列特征分支: 准确率={test_accuracy_seq:.4f}, 损失={test_loss_seq:.4f}")
        print(f"  特征融合分支: 准确率={test_accuracy_fused:.4f}, 损失={test_loss_fused:.4f}")
        print(f"  分支平均准确率: {(test_accuracy_stats + test_accuracy_seq) / 2:.4f}")

        # 更新最佳分支准确率记录
        if test_accuracy_stats > self.best_stats_accuracy:
            self.best_stats_accuracy = test_accuracy_stats
            print(f"  📈 统计分支最佳准确率更新至: {self.best_stats_accuracy:.4f}")
        if test_accuracy_seq > self.best_seq_accuracy:
            self.best_seq_accuracy = test_accuracy_seq
            print(f"  📈 序列分支最佳准确率更新至: {self.best_seq_accuracy:.4f}")

        if test_accuracy_fused > self.best_accuracy:
            print(f"融合分支准确率提升至 {test_accuracy_fused:.4f}，保存模型...")
            save_dir = os.path.dirname(self.model_save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': test_accuracy_fused,
                'best_stats_accuracy': self.best_stats_accuracy,
                'best_seq_accuracy': self.best_seq_accuracy,
                'stats_accuracy': test_accuracy_stats,
                'seq_accuracy': test_accuracy_seq,
                'stats_loss': test_loss_stats,
                'seq_loss': test_loss_seq,
                'fused_loss': test_loss_fused
            }
            if center_optimizer is not None:
                save_dict['center_optimizer_state_dict'] = center_optimizer.state_dict()

            torch.save(save_dict, self.model_save_path)
            self.best_accuracy = test_accuracy_fused
            self.wait = 0
        else:
            self.wait += 1
            print(f"测试集准确率未提升，等待次数: {self.wait}/{self.patience}")

        return test_accuracy_fused, test_accuracy_stats, test_accuracy_seq


# 核心修改：增强的训练函数，包含分支损失
def train_model(model, X_stats_train, X_seqs_train, y_train,
                X_stats_test, X_seqs_test, y_test,
                epochs=50, batch_size=8,
                accumulation_steps=2,
                model_save_path="best_innovative_model.pth",
                patience=15, learning_rate=1e-4,
                loss_weights={'stats': 0.3, 'seq': 0.3, 'fused': 0.4, 'center': 0.2}):
    """训练函数（核心优化：增加分支损失 + 中心损失训练策略）"""
    print(f"训练集类别分布: {Counter(np.argmax(y_train, axis=1))}")
    print(f"训练集大小: {X_stats_train.shape[0]} 样本")
    print(
        f"训练策略：Batch Size={batch_size} + 梯度累积*{accumulation_steps}（有效Batch Size={batch_size * accumulation_steps}）")
    print(
        f"损失权重配置：统计分支={loss_weights['stats']}, 序列分支={loss_weights['seq']}, 融合分支={loss_weights['fused']}, 中心损失={loss_weights['center']}")

    y_train_onehot = y_train
    num_classes = y_train_onehot.shape[1]

    # 创建数据加载器（保持不变）
    train_dataset = TensorDataset(
        torch.FloatTensor(X_stats_train),
        torch.FloatTensor(X_seqs_train),
        torch.FloatTensor(y_train_onehot)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=0,
        drop_last=True
    )

    # 优化器（核心改进：为中心损失单独设置优化器和学习率）
    optimizer = optim.AdamW([
        {'params': model.parameters(), 'lr': learning_rate}
    ], weight_decay=0.001)

    # 补充：定义学习率调度器（修复scheduler未定义的问题）
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # 中心损失单独优化器（学习率通常为主学习率的1/10 ~ 1/100）
    criterion_center = ImprovedCenterLoss(
        num_classes=num_classes,
        feat_dim=model.stats_output_dim,
        device=device,
        alpha=0.95  # 动量系数（0.9-0.99）
    )
    center_optimizer = optim.AdamW([
        {'params': criterion_center.parameters(), 'lr': learning_rate * 0.05}  # 中心学习率=主学习率×0.05
    ], weight_decay=0.0001)  # 更小的权重衰减

    # 动态权重调度器（核心改进：随着训练调整各损失权重）
    def get_dynamic_loss_weights(epoch, warmup_epochs=15):
        """
        动态损失权重策略：
        1. 前warmup_epochs轮：逐步增加分支损失权重，确保分支先收敛
        2. 之后：保持设定的权重比例
        """
        if epoch < warmup_epochs:
            # 热身阶段：增加分支损失权   重，降低融合损失权重
            warmup_factor = epoch / warmup_epochs
            return {
                'stats': loss_weights['stats'] * warmup_factor + 0.1 * (1 - warmup_factor),
                'seq': loss_weights['seq'] * warmup_factor + 0.1 * (1 - warmup_factor),
                'fused': loss_weights['fused'] * (1 - warmup_factor) + 0.6 * warmup_factor,
                'center': loss_weights['center']
            }
        else:
            return loss_weights

    criterion_cls = nn.CrossEntropyLoss()

    # 回调函数
    test_callback = TestAccuracyCallback(
        model_save_path=model_save_path,
        X_stats_test=X_stats_test,
        X_seqs_test=X_seqs_test,
        y_test=y_test,
        patience=patience,
        eval_batch_size=64
    )

    # 训练循环（优化中心损失训练流程 + 分支损失）
    history = {
        'accuracy': [], 'stats_accuracy': [], 'seq_accuracy': [],
        'loss': [], 'stats_loss': [], 'seq_loss': [], 'fused_loss': [], 'center_loss': [], 'total_loss': [],
        'center_weight': [], 'loss_weights': []
    }

    print(f"\n开始训练：{epochs}个epoch，批次大小{batch_size}，梯度累积×{accumulation_steps}")
    print(f"学习率：主学习率={learning_rate}，中心学习率={learning_rate * 0.05}")
    print(f"分支损失配置：统计+序列+融合+中心损失的多任务学习")

    for epoch in range(epochs):
        model.train()
        criterion_center.train()  # 确保中心损失处于训练模式
        epoch_loss_stats = 0.0
        epoch_loss_seq = 0.0
        epoch_loss_fused = 0.0
        epoch_loss_center = 0.0
        epoch_total_loss = 0.0
        correct = 0
        correct_stats = 0
        correct_seq = 0
        total = 0

        # 获取当前epoch的动态损失权重
        dynamic_weights = get_dynamic_loss_weights(epoch)
        history['loss_weights'].append(dynamic_weights)
        history['center_weight'].append(dynamic_weights['center'])

        optimizer.zero_grad()
        center_optimizer.zero_grad()

        for batch_idx, (batch_stats, batch_seqs, batch_labels) in enumerate(train_loader):
            batch_stats = batch_stats.to(device, non_blocking=GPU_AVAILABLE)
            batch_seqs = batch_seqs.to(device, non_blocking=GPU_AVAILABLE)
            batch_labels = batch_labels.to(device, non_blocking=GPU_AVAILABLE)
            batch_labels_int = torch.argmax(batch_labels, dim=1)

            try:
                # 前向传播（获取统计特征用于中心损失）
                stats_output, seq_output, fused_output, _, stats_feat = model(
                    batch_stats, batch_seqs, return_branches=True, return_attn_weights=True
                )

                # 初始化中心（仅在第一个batch执行）
                criterion_center.initialize_centers(stats_feat, batch_labels_int)

            except Exception as e:
                print(f"\n❌ 第{epoch + 1}epoch第{batch_idx + 1}批次训练失败：{str(e)}")
                print(f"  输入维度：stats={batch_stats.shape}, seqs={batch_seqs.shape}")
                continue

            # 计算各分支损失
            loss_stats = criterion_cls(stats_output, batch_labels_int) / accumulation_steps
            loss_seq = criterion_cls(seq_output, batch_labels_int) / accumulation_steps
            loss_fused = criterion_cls(fused_output, batch_labels_int) / accumulation_steps
            loss_center = criterion_center(stats_feat, batch_labels_int) / accumulation_steps

            # 加权总损失
            total_loss = (
                    dynamic_weights['stats'] * loss_stats +
                    dynamic_weights['seq'] * loss_seq +
                    dynamic_weights['fused'] * loss_fused +
                    dynamic_weights['center'] * loss_center
            )

            # 反向传播
            total_loss.backward()

            # 统计损失
            epoch_loss_stats += loss_stats.item() * accumulation_steps
            epoch_loss_seq += loss_seq.item() * accumulation_steps
            epoch_loss_fused += loss_fused.item() * accumulation_steps
            epoch_loss_center += loss_center.item() * accumulation_steps
            epoch_total_loss += total_loss.item() * accumulation_steps

            # 统计各分支准确率
            _, predicted_stats = torch.max(stats_output.data, 1)
            _, predicted_seq = torch.max(seq_output.data, 1)
            _, predicted = torch.max(fused_output.data, 1)

            total += batch_labels.size(0)
            correct += (predicted == batch_labels_int).sum().item()
            correct_stats += (predicted_stats == batch_labels_int).sum().item()
            correct_seq += (predicted_seq == batch_labels_int).sum().item()

            # 梯度累积更新
            if (batch_idx + 1) % accumulation_steps == 0:
                # 主模型参数更新
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # 中心参数单独更新（关键：分开更新，避免梯度冲突）
                torch.nn.utils.clip_grad_norm_(criterion_center.parameters(), max_norm=0.5)
                center_optimizer.step()

                # 清零梯度
                optimizer.zero_grad()
                center_optimizer.zero_grad()

            # 打印日志
            if (batch_idx + 1) % 100 == 0:
                batch_accuracy = correct / total
                batch_accuracy_stats = correct_stats / total
                batch_accuracy_seq = correct_seq / total

                batch_loss_stats = epoch_loss_stats / (batch_idx + 1)
                batch_loss_seq = epoch_loss_seq / (batch_idx + 1)
                batch_loss_fused = epoch_loss_fused / (batch_idx + 1)
                batch_loss_center = epoch_loss_center / (batch_idx + 1)
                batch_total_loss = epoch_total_loss / (batch_idx + 1)

                print(f"  Epoch {epoch + 1}/{epochs} | Batch {batch_idx + 1}/{len(train_loader)} | "
                      f"融合准确率: {batch_accuracy:.4f} | 统计准确率: {batch_accuracy_stats:.4f} | 序列准确率: {batch_accuracy_seq:.4f} | "
                      f"统计损失: {batch_loss_stats:.4f} | 序列损失: {batch_loss_seq:.4f} | 融合损失: {batch_loss_fused:.4f} | "
                      f"中心损失: {batch_loss_center:.4f} | 总损失: {batch_total_loss:.4f}")

        scheduler.step()  # 现在scheduler已定义，可正常调用

        # 计算epoch平均指标
        train_accuracy = correct / total if total > 0 else 0.0
        train_accuracy_stats = correct_stats / total if total > 0 else 0.0
        train_accuracy_seq = correct_seq / total if total > 0 else 0.0

        train_loss_stats = epoch_loss_stats / len(train_loader) if len(train_loader) > 0 else 0.0
        train_loss_seq = epoch_loss_seq / len(train_loader) if len(train_loader) > 0 else 0.0
        train_loss_fused = epoch_loss_fused / len(train_loader) if len(train_loader) > 0 else 0.0
        train_loss_center = epoch_loss_center / len(train_loader) if len(train_loader) > 0 else 0.0
        train_total_loss = epoch_total_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        print(f"\nEpoch {epoch + 1} 训练集总结:")
        print(
            f"  融合准确率={train_accuracy:.4f}, 统计准确率={train_accuracy_stats:.4f}, 序列准确率={train_accuracy_seq:.4f}")
        print(f"  统计损失={train_loss_stats:.4f}, 序列损失={train_loss_seq:.4f}, 融合损失={train_loss_fused:.4f}")
        print(f"  中心损失={train_loss_center:.4f}, 总损失={train_total_loss:.4f}")
        print(
            f"  动态损失权重：统计={dynamic_weights['stats']:.3f}, 序列={dynamic_weights['seq']:.3f}, 融合={dynamic_weights['fused']:.3f}, 中心={dynamic_weights['center']:.3f}")

        # 记录历史
        history['accuracy'].append(train_accuracy)
        history['stats_accuracy'].append(train_accuracy_stats)
        history['seq_accuracy'].append(train_accuracy_seq)
        history['stats_loss'].append(train_loss_stats)
        history['seq_loss'].append(train_loss_seq)
        history['fused_loss'].append(train_loss_fused)
        history['center_loss'].append(train_loss_center)
        history['loss'].append(train_total_loss)  # 兼容旧的loss字段
        history['total_loss'].append(train_total_loss)

        try:
            test_accuracy_fused, test_accuracy_stats, test_accuracy_seq = test_callback.on_epoch_end(epoch, model,
                                                                                                     optimizer,
                                                                                                     center_optimizer)
            # 记录测试集准确率
            if 'test_accuracy' not in history:
                history['test_accuracy'] = []
                history['test_stats_accuracy'] = []
                history['test_seq_accuracy'] = []
            history['test_accuracy'].append(test_accuracy_fused)
            history['test_stats_accuracy'].append(test_accuracy_stats)
            history['test_seq_accuracy'].append(test_accuracy_seq)

        except Exception as e:
            print(f"⚠️ 测试集评估失败：{str(e)}")
            test_accuracy_fused = 0.0

        # 主动清理显存
        torch.cuda.empty_cache()
        gc.collect()

        if test_callback.wait >= test_callback.patience:
            print(f"\n训练在 epoch {epoch + 1} 提前停止")
            print(
                f"最佳测试集准确率: 融合={test_callback.best_accuracy:.4f}, 统计={test_callback.best_stats_accuracy:.4f}, 序列={test_callback.best_seq_accuracy:.4f}")
            break

    del train_dataset, train_loader
    gc.collect()

    return history


# --------------------------
# 评估和可视化模块（增强分支损失分析）
# --------------------------
def evaluate_model_with_branches(model, X_stats_test, X_seqs_test, y_test, class_names):
    """评估函数（增强分支评估）"""
    model.eval()

    batch_size_eval = 64
    n_samples_test = X_stats_test.shape[0]
    n_batches_eval = (n_samples_test + batch_size_eval - 1) // batch_size_eval

    y_pred_stats_all = []
    y_pred_seq_all = []
    y_pred_fused_all = []
    y_true_all = []

    # 计算各分支损失
    criterion_cls = nn.CrossEntropyLoss()
    total_loss_stats = 0.0
    total_loss_seq = 0.0
    total_loss_fused = 0.0

    with torch.no_grad():
        for batch_idx in range(n_batches_eval):
            start_idx = batch_idx * batch_size_eval
            end_idx = min((batch_idx + 1) * batch_size_eval, n_samples_test)

            stats_batch = torch.FloatTensor(X_stats_test[start_idx:end_idx]).to(device)
            seqs_batch = torch.FloatTensor(X_seqs_test[start_idx:end_idx]).to(device)
            y_batch = torch.FloatTensor(y_test[start_idx:end_idx]).to(device)
            y_batch_int = torch.argmax(y_batch, dim=1)

            stats_output, seq_output, fused_output = model(stats_batch, seqs_batch, return_branches=True)

            # 计算损失
            loss_stats = criterion_cls(stats_output, y_batch_int)
            loss_seq = criterion_cls(seq_output, y_batch_int)
            loss_fused = criterion_cls(fused_output, y_batch_int)

            total_loss_stats += loss_stats.item() * len(y_batch)
            total_loss_seq += loss_seq.item() * len(y_batch)
            total_loss_fused += loss_fused.item() * len(y_batch)

            y_pred_stats = torch.argmax(stats_output, dim=1).cpu().numpy()
            y_pred_seq = torch.argmax(seq_output, dim=1).cpu().numpy()
            y_pred_fused = torch.argmax(fused_output, dim=1).cpu().numpy()
            y_true = y_batch_int.cpu().numpy()

            y_pred_stats_all.extend(y_pred_stats)
            y_pred_seq_all.extend(y_pred_seq)
            y_pred_fused_all.extend(y_pred_fused)
            y_true_all.extend(y_true)

    y_pred_stats = np.array(y_pred_stats_all)
    y_pred_seq = np.array(y_pred_seq_all)
    y_pred_fused = np.array(y_pred_fused_all)
    y_true = np.array(y_true_all)

    # 计算准确率和损失
    accuracy_stats = accuracy_score(y_true, y_pred_stats)
    accuracy_seq = accuracy_score(y_true, y_pred_seq)
    accuracy_fused = accuracy_score(y_true, y_pred_fused)

    loss_stats = total_loss_stats / n_samples_test
    loss_seq = total_loss_seq / n_samples_test
    loss_fused = total_loss_fused / n_samples_test

    print("\n" + "=" * 80)
    print("📊 各分支详细评估报告（测试集）- 多分支损失优化版")
    print("=" * 80)
    print(f"统计特征分支: 准确率={accuracy_stats:.4f}, 损失={loss_stats:.4f}")
    print(f"序列特征分支: 准确率={accuracy_seq:.4f}, 损失={loss_seq:.4f}")
    print(f"特征融合分支: 准确率={accuracy_fused:.4f}, 损失={loss_fused:.4f}")
    print(f"融合提升: {accuracy_fused - max(accuracy_stats, accuracy_seq):.4f}")
    print(f"分支平均准确率: {(accuracy_stats + accuracy_seq) / 2:.4f}")

    # 各分支详细分类报告
    print("\n=== 统计特征分支详细分类报告 ===")
    print(classification_report(y_true, y_pred_stats, target_names=class_names))

    print("\n=== 序列特征分支详细分类报告 ===")
    print(classification_report(y_true, y_pred_seq, target_names=class_names))

    print("\n=== 融合分支详细分类报告 ===")
    report = classification_report(y_true, y_pred_fused, target_names=class_names, output_dict=True)
    print(classification_report(y_true, y_pred_fused, target_names=class_names))

    print("\n各类别详细指标（融合分支）:")
    for cls in class_names:
        print(f"{cls}:")
        print(f"  精确率: {report[cls]['precision']:.4f}")
        print(f"  召回率: {report[cls]['recall']:.4f}")
        print(f"  F1分数: {report[cls]['f1-score']:.4f}")
        print(f"  支持数: {report[cls]['support']}")

    # 绘制三个分支的混淆矩阵
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # 统计分支混淆矩阵
    cm_stats = confusion_matrix(y_true, y_pred_stats)
    sns.heatmap(cm_stats, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 10}, ax=axes[0])
    axes[0].set_xlabel('预测类别', fontsize=12)
    axes[0].set_ylabel('真实类别', fontsize=12)
    axes[0].set_title(f'统计特征分支混淆矩阵 (准确率={accuracy_stats:.4f})', fontsize=14)

    # 序列分支混淆矩阵
    cm_seq = confusion_matrix(y_true, y_pred_seq)
    sns.heatmap(cm_seq, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 10}, ax=axes[1])
    axes[1].set_xlabel('预测类别', fontsize=12)
    axes[1].set_ylabel('真实类别', fontsize=12)
    axes[1].set_title(f'序列特征分支混淆矩阵 (准确率={accuracy_seq:.4f})', fontsize=14)

    # 融合分支混淆矩阵
    cm_fused = confusion_matrix(y_true, y_pred_fused)
    sns.heatmap(cm_fused, annot=True, fmt='d', cmap='Reds',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 10}, ax=axes[2])
    axes[2].set_xlabel('预测类别', fontsize=12)
    axes[2].set_ylabel('真实类别', fontsize=12)
    axes[2].set_title(f'融合分支混淆矩阵 (准确率={accuracy_fused:.4f})', fontsize=14)

    plt.tight_layout()
    plt.savefig('confusion_matrices_branches.png', dpi=300, bbox_inches='tight')
    print("三个分支的混淆矩阵已保存")

    return y_pred_fused, y_true, {'stats': accuracy_stats, 'seq': accuracy_seq, 'fused': accuracy_fused}


def plot_training_history(history):
    """训练曲线绘制（增强分支损失和准确率可视化）"""
    plt.figure(figsize=(20, 15))

    # 准确率曲线（训练集）
    plt.subplot(3, 3, 1)
    plt.plot(history['accuracy'], label='融合分支准确率', linewidth=2, color='#2E86AB')
    plt.plot(history['stats_accuracy'], label='统计分支准确率', linewidth=2, color='#A23B72')
    plt.plot(history['seq_accuracy'], label='序列分支准确率', linewidth=2, color='#F18F01')
    plt.grid(True, alpha=0.3)
    plt.title('训练集各分支准确率变化', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('准确率', fontsize=10)
    plt.legend(fontsize=9)

    # 准确率曲线（测试集）
    if 'test_accuracy' in history:
        plt.subplot(3, 3, 2)
        plt.plot(history['test_accuracy'], label='融合分支准确率', linewidth=2, color='#2E86AB')
        plt.plot(history['test_stats_accuracy'], label='统计分支准确率', linewidth=2, color='#A23B72')
        plt.plot(history['test_seq_accuracy'], label='序列分支准确率', linewidth=2, color='#F18F01')
        plt.grid(True, alpha=0.3)
        plt.title('测试集各分支准确率变化', fontsize=12)
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('准确率', fontsize=10)
        plt.legend(fontsize=9)

    # 各分支损失曲线
    plt.subplot(3, 3, 3)
    plt.plot(history['stats_loss'], label='统计分支损失', linewidth=2, color='#A23B72')
    plt.plot(history['seq_loss'], label='序列分支损失', linewidth=2, color='#F18F01')
    plt.plot(history['fused_loss'], label='融合分支损失', linewidth=2, color='#2E86AB')
    plt.plot(history['center_loss'], label='中心损失', linewidth=2, color='#6A994E')
    plt.grid(True, alpha=0.3)
    plt.title('各分支损失变化曲线', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('损失', fontsize=10)
    plt.legend(fontsize=9)

    # 总损失曲线
    plt.subplot(3, 3, 4)
    plt.plot(history['total_loss'], label='总损失', linewidth=2, color='#C73E1D')
    plt.grid(True, alpha=0.3)
    plt.title('总损失变化曲线', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('损失', fontsize=10)
    plt.legend(fontsize=9)

    # 中心损失（对数尺度）
    plt.subplot(3, 3, 5)
    plt.plot(history['center_loss'], label='中心损失（log）', linewidth=2, color='#6A994E')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.title('中心损失变化曲线（对数尺度）', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('损失（log）', fontsize=10)
    plt.legend(fontsize=9)

    # 统计分支：准确率 vs 损失
    plt.subplot(3, 3, 6)
    ax1 = plt.gca()
    ax1.plot(history['stats_accuracy'], label='统计分支准确率', linewidth=2, color='#A23B72')
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('准确率', fontsize=10, color='#A23B72')
    ax1.tick_params(axis='y', labelcolor='#A23B72')

    ax2 = ax1.twinx()
    ax2.plot(history['stats_loss'], label='统计分支损失', linewidth=2, color='#C73E1D', linestyle='--')
    ax2.set_ylabel('损失', fontsize=10, color='#C73E1D')
    ax2.tick_params(axis='y', labelcolor='#C73E1D')

    plt.title('统计分支：准确率 vs 损失', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3)

    # 序列分支：准确率 vs 损失
    plt.subplot(3, 3, 7)
    ax1 = plt.gca()
    ax1.plot(history['seq_accuracy'], label='序列分支准确率', linewidth=2, color='#F18F01')
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('准确率', fontsize=10, color='#F18F01')
    ax1.tick_params(axis='y', labelcolor='#F18F01')

    ax2 = ax1.twinx()
    ax2.plot(history['seq_loss'], label='序列分支损失', linewidth=2, color='#C73E1D', linestyle='--')
    ax2.set_ylabel('损失', fontsize=10, color='#C73E1D')
    ax2.tick_params(axis='y', labelcolor='#C73E1D')

    plt.title('序列分支：准确率 vs 损失', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3)

    # 融合分支：准确率 vs 损失
    plt.subplot(3, 3, 8)
    ax1 = plt.gca()
    ax1.plot(history['accuracy'], label='融合分支准确率', linewidth=2, color='#2E86AB')
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('准确率', fontsize=10, color='#2E86AB')
    ax1.tick_params(axis='y', labelcolor='#2E86AB')

    ax2 = ax1.twinx()
    ax2.plot(history['fused_loss'], label='融合分支损失', linewidth=2, color='#C73E1D', linestyle='--')
    ax2.set_ylabel('损失', fontsize=10, color='#C73E1D')
    ax2.tick_params(axis='y', labelcolor='#C73E1D')

    plt.title('融合分支：准确率 vs 损失', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3)

    # 损失权重变化（如果有）
    if 'loss_weights' in history and len(history['loss_weights']) > 0:
        plt.subplot(3, 3, 9)
        stats_weights = [w['stats'] for w in history['loss_weights']]
        seq_weights = [w['seq'] for w in history['loss_weights']]
        fused_weights = [w['fused'] for w in history['loss_weights']]
        center_weights = [w['center'] for w in history['loss_weights']]

        plt.plot(stats_weights, label='统计损失权重', linewidth=2, color='#A23B72')
        plt.plot(seq_weights, label='序列损失权重', linewidth=2, color='#F18F01')
        plt.plot(fused_weights, label='融合损失权重', linewidth=2, color='#2E86AB')
        plt.plot(center_weights, label='中心损失权重', linewidth=2, color='#6A994E')
        plt.grid(True, alpha=0.3)
        plt.title('动态损失权重变化', fontsize=12)
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('权重值', fontsize=10)
        plt.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('training_metrics_branches_optimized.png', dpi=300, bbox_inches='tight')
    print("增强版训练指标曲线已保存（包含分支损失和准确率分析）")


# --------------------------
# 主程序入口（增强分支损失配置）
# --------------------------
if __name__ == "__main__":
    try:
        # 配置参数
        BATCH_SIZE = 1024
        ACCUMULATION_STEPS = 2
        EPOCHS = 400
        PATIENCE = 60
        MODEL_SAVE_PATH = r"best_branches_optimized_model.pth"
        NPZ_FILE_PATH = r"traffic_data_zuni——12.11.npz"

        # 分支损失权重配置
        LOSS_WEIGHTS = {
            'stats': 0.3,  # 统计特征分支损失权重
            'seq': 0.3,  # 序列特征分支损失权重
            'fused': 0.3,  # 融合分支损失权重
            'center': 0.1  # 中心损失权重
        }

        print("=" * 80)
        print("🚀 开始从NPZ文件加载数据进行训练（多分支损失优化版）")
        print("=" * 80)
        print(f"⚠️  配置：Batch Size={BATCH_SIZE} + 梯度累积×{ACCUMULATION_STEPS} + 70%显存限制")
        print(
            f"🔧 多分支损失优化：统计分支({LOSS_WEIGHTS['stats']}) + 序列分支({LOSS_WEIGHTS['seq']}) + 融合分支({LOSS_WEIGHTS['fused']}) + 中心损失({LOSS_WEIGHTS['center']})")
        print(f"🔧 动态权重策略：前15轮热身，逐步调整各分支损失权重")

        if not os.path.exists(NPZ_FILE_PATH):
            raise FileNotFoundError(f"NPZ文件不存在：{NPZ_FILE_PATH}")

        print(f"✅ 正在加载NPZ文件：{NPZ_FILE_PATH}")
        cache_data = np.load(NPZ_FILE_PATH, allow_pickle=True)

        # 兼容两种NPZ命名：
        # 1) 旧版预处理脚本保存 label_encoder
        # 2) 新版 data_preprocessing.py 保存 classes
        required_keys = ["X_stats_train", "X_seqs_train", "y_train",
                         "X_stats_test", "X_seqs_test", "y_test"]

        missing_keys = [key for key in required_keys if key not in cache_data]
        if missing_keys:
            raise ValueError(f"NPZ文件缺少必要数据：{missing_keys}")

        has_label_encoder = "label_encoder" in cache_data
        has_classes = "classes" in cache_data
        has_class_names = "class_names" in cache_data

        if not (has_label_encoder or has_classes or has_class_names):
            raise ValueError(
                "NPZ文件缺少类别元数据：需要包含 label_encoder、classes 或 class_names 其中之一。"
                "如果使用新版 data_preprocessing.py，请确认保存了 classes 字段。"
            )

        print("✅ NPZ文件验证通过")

        # 提取数据
        X_stats_train = cache_data["X_stats_train"].astype(np.float32)
        X_seqs_train = cache_data["X_seqs_train"].astype(np.float32)
        y_train = cache_data["y_train"].astype(np.int64)

        X_stats_test = cache_data["X_stats_test"].astype(np.float32)
        X_seqs_test = cache_data["X_seqs_test"].astype(np.float32)
        y_test = cache_data["y_test"].astype(np.int64)

        # 统一类别名称读取逻辑
        if has_label_encoder:
            label_encoder = cache_data["label_encoder"].item()
            class_names = np.asarray(label_encoder.classes_)
            print("✅ 使用 NPZ 中的 label_encoder 读取类别名称")
        elif has_classes:
            class_names = np.asarray(cache_data["classes"])
            print("✅ 使用 NPZ 中的 classes 读取类别名称")
        else:
            class_names = np.asarray(cache_data["class_names"])
            print("✅ 使用 NPZ 中的 class_names 读取类别名称")

        # 防止 classes 被保存成 0 维 object array
        if class_names.shape == ():
            class_names = np.asarray(class_names.item())

        class_names = class_names.astype(str)
        num_classes = len(class_names)

        # 基础一致性检查，避免标签编号和类别数量不匹配
        max_label = int(max(np.max(y_train), np.max(y_test)))
        if max_label >= num_classes:
            raise ValueError(
                f"标签编号和类别数量不匹配：最大标签编号={max_label}, 类别数量={num_classes}, "
                f"类别名称={class_names}"
            )

        print(f"\n📊 数据集统计:")
        print(f"  训练集: {X_stats_train.shape[0]} 样本")
        print(f"  测试集: {X_stats_test.shape[0]} 样本")
        print(f"  统计特征维度: {X_stats_train.shape[1]} 维")
        print(f"  序列特征形状: {X_seqs_train.shape[1:]}")
        print(f"  类别数: {num_classes} ({class_names})")
        print(f"  训练集类别分布: {dict(zip(class_names, np.bincount(y_train, minlength=num_classes)))}")
        print(f"  测试集类别分布: {dict(zip(class_names, np.bincount(y_test, minlength=num_classes)))}")

        # 转换为one-hot编码

        # 转换为one-hot编码
        y_train_onehot = np.eye(num_classes)[y_train].astype(np.float32)
        y_test_onehot = np.eye(num_classes)[y_test].astype(np.float32)

        # 构建模型
        stats_shape = X_stats_train.shape[1:]
        seq_shape = X_seqs_train.shape[1:]
        model = create_gru_first_model(stats_shape, seq_shape, num_classes)

        # 训练模型（使用分支损失）
        print("\n🎯 开始多分支损失优化的GRU-TCN+自注意力+改进版中心损失模型训练...")
        history = train_model(
            model,
            X_stats_train, X_seqs_train, y_train_onehot,
            X_stats_test, X_seqs_test, y_test_onehot,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            accumulation_steps=ACCUMULATION_STEPS,
            model_save_path=MODEL_SAVE_PATH,
            patience=PATIENCE,
            learning_rate=1e-4,
            loss_weights=LOSS_WEIGHTS
        )

        # 加载最佳模型
        print(f"\n📈 加载最佳模型进行最终评估...")
        if os.path.exists(MODEL_SAVE_PATH):
            checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"加载 epoch {checkpoint['epoch']} 的最佳模型")
            print(
                f"最佳测试集准确率: 融合={checkpoint['best_accuracy']:.4f}, 统计={checkpoint['best_stats_accuracy']:.4f}, 序列={checkpoint['best_seq_accuracy']:.4f}")
        else:
            print("⚠️ 未找到最佳模型文件，使用最后一个epoch的模型进行评估")

        # 详细评估
        print("\n📈 多分支损失优化模型详细分类报告:")
        y_pred_fused, y_true, accuracies = evaluate_model_with_branches(model, X_stats_test, X_seqs_test, y_test_onehot,
                                                                        class_names)

        plot_training_history(history)

        print(f"\n✅ 训练完成（多分支损失优化版）！最佳模型已保存至: {MODEL_SAVE_PATH}")
        print(f"🔧 多分支损失优化总结:")
        print(f"  1. 新增统计特征分支损失：权重={LOSS_WEIGHTS['stats']}，针对性提升统计分支准确率")
        print(f"  2. 新增序列特征分支损失：权重={LOSS_WEIGHTS['seq']}，针对性提升序列分支准确率")
        print(f"  3. 动态权重策略：前15轮热身，确保各分支先收敛再融合")
        print(f"  4. 中心损失优化：类别均值初始化 + 单独优化器 + 动量更新")
        print(
            f"  5. 最终准确率：统计={accuracies['stats']:.4f}, 序列={accuracies['seq']:.4f}, 融合={accuracies['fused']:.4f}")

    except Exception as e:
        print(f"\n❌ 程序运行失败：{str(e)}")
        import traceback

        traceback.print_exc()