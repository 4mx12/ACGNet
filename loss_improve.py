import torch
import torch.nn as nn
import torch.nn.functional as F



class AdaptiveLaplacianEdgeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化4个方向的可学习拉普拉斯核
        self.kernels = nn.Parameter(torch.randn(4,1,3,3))  # [out_channels=4, in_channels=1, H, W]
        nn.init.constant_(self.kernels, 0.0)
        # 预定义基础方向模板
        self.kernels.data[0] = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]])/4.0  # 0度
        self.kernels.data[1] = torch.tensor([[1,0,1],[0,-4,0],[1,0,1]])/4.0  # 45度
        self.kernels.data[2] = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]]).T/4.0  # 90度
        self.kernels.data[3] = torch.tensor([[1,0,1],[0,-4,0],[1,0,1]]).T/4.0  # 135度

    def forward(self, x):
        # x: [B,1,H,W]
        # x_log = torch.log(x + 1e-6)
        # 多方向卷积（输出通道=4）
        edge_maps = F.conv2d(x, self.kernels, padding=1, groups=1)  # shape [B,4,H,W]
        # 动态加权融合
        return edge_maps.mean(dim=1, keepdim=True)  # [B,1,H,W]






class AdaptiveTVLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.avg_pool = nn.AvgPool2d(3, stride=1, padding=1)  # 新增平均池化

    def _local_entropy(self, x):
        # 保持输入尺寸的池化操作
        x_pad = F.pad(x, (1, 1, 1, 1), mode='reflect')  # 反射填充避免边缘效应
        unfolded = x_pad.unfold(2, 3, 1).unfold(3, 3, 1)  # [B,C,H,W,3,3]
        var = unfolded.var(dim=(4, 5), keepdim=True)  # 计算局部方差
        entropy = - (var * torch.log(var + self.eps)).sum(dim=(4, 5))  # [B,C,H,W]
        return self.avg_pool(entropy)  # 平滑熵图

    def forward(self, x):
        # 水平/垂直差分
        diff_h = x[:, :, :, 1:] - x[:, :, :, :-1]  # [B,C,H,W-1]
        diff_v = x[:, :, 1:, :] - x[:, :, :-1, :]  # [B,C,H-1,W]

        # 熵权重计算
        entropy = self._local_entropy(x)
        weights_h = 1 / (entropy[:, :, :, :-1] + 0.1)  # 对齐W-1
        weights_v = 1 / (entropy[:, :, :-1, :] + 0.1)  # 对齐H-1

        # 加权损失
        loss_h = (diff_h.abs() * weights_h).mean()
        loss_v = (diff_v.abs() * weights_v).mean()
        return (loss_h + loss_v) / 2.0


class SARLoss(nn.Module):
    
    def __init__(self, lambda_edge=0.5, lambda_tv=0.005, print_interval=50):
        super().__init__()
        self.lambda_edge = lambda_edge
        self.lambda_tv = lambda_tv
        self.print_interval = print_interval  # 打印间隔步数
        self.edge_dec = AdaptiveLaplacianEdgeDetector()
        self.tv_loss = AdaptiveTVLoss()
        self.mse = nn.MSELoss()
        self.step = 0  # 训练步骤计数器

    def forward(self, denoised, clean):
        # 计算各项损失
        mse_loss = self.mse(denoised, clean)
        edge_loss = self.mse(self.edge_dec(denoised), self.edge_dec(clean))
        tv_loss = self.tv_loss(denoised)
        total_loss = mse_loss + self.lambda_edge * edge_loss + self.lambda_tv * tv_loss

        # 训练模式下打印日志
        if self.training:
            self.step += 1
            if self.step % self.print_interval == 0:
                print(f"[Step {self.step}] "
                      f"MSE: {mse_loss.item():.4f} | "
                      f"Edge: {edge_loss.item():.4f} | "
                      f"TV: {tv_loss.item():.4f} | "
                      f"Total: {total_loss.item():.4f}")

        return total_loss

    def extra_repr(self):
        """在print(model)时显示当前权重配置"""
        return f"λ_edge={self.lambda_edge}, λ_tv={self.lambda_tv}"


def test_dimension_alignment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"测试设备：{device}")

    # 生成模拟数据 (batch=2, 1通道, 64x64)
    x = torch.rand(2, 1, 64, 64, device=device).requires_grad_(True)
    tv_loss = AdaptiveTVLoss().to(device)

    # 前向计算
    loss = tv_loss(x)

    # 维度检查
    print("输入尺寸:", x.shape)  # [2,1,64,64]
    entropy = tv_loss._local_entropy(x)
    print("局部熵尺寸:", entropy.shape)  # [2,1,64,64]

    diff_h = x[:, :, :, 1:] - x[:, :, :, :-1]
    weights_h = 1 / (entropy[:, :, :, :-1] + 0.1)
    print("水平梯度尺寸:", diff_h.shape)  # [2,1,64,63]
    print("水平权重尺寸:", weights_h.shape)  # [2,1,64,63]

    # 反向传播测试
    loss.backward()
    print("梯度计算成功！")


if __name__ == "__main__":
    # 执行测试
    torch.manual_seed(42)
    test_dimension_alignment()
