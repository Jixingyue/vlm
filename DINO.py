"""
DINO（self-DIstillation with NO labels，无标签自蒸馏）的最小实现示例。

核心思想（小白版）：
    这是一种自监督学习（不需要人工标签）。有两个结构相同的网络：
      - 学生网络 gs（student）：通过梯度下降正常训练更新；
      - 教师网络 gt（teacher）：不用梯度，而是用“动量”方式缓慢地拷贝学生的参数（相当于学生的平滑版本）。
    对同一张图片做两次不同的随机增强（x1、x2），让学生和教师分别去处理，
    训练目标是：让“学生的输出”去靠近“教师的输出”（交叉视图一致）。

    两个关键设计：
      1) centering（中心化，用变量 C）：防止所有输出都崩塌到同一个点（避免模型崩溃）；
      2) sharpening（锐化，用温度 tps/tpt）：让输出分布更尖锐，与 centering 互相平衡。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# Set up device
# 自动选择计算设备：有 GPU 就用 cuda，否则用 cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Student and Teacher networks using a Vision Transformer (ViT) architecture
# 学生和教师网络都使用相同结构的 ViT
class ViT(nn.Module):
    """ViT 图像编码器：输入图片，输出一个 output_dim 维的特征向量。"""
    def __init__(self, output_dim):
        super(ViT, self).__init__()
        # Use a ViT model from timm
        # pretrained=False：DINO 是从头开始自监督训练，不需要预训练权重
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=output_dim)

    def forward(self, x):
        return self.vit(x)

# Data loading and augmentation
def get_dataloader(batch_size):
    """加载 CIFAR10 并定义数据增强（自监督学习不关心标签，只关心图像本身）。"""
    transform = Compose([
        Resize(224),  # Resize the image to 224x224，缩放到 ViT 要求的尺寸
        RandomCrop(224, padding=4),   # 随机裁剪：先填充 4 像素再裁剪，制造轻微位移
        RandomHorizontalFlip(),        # 随机水平翻转
        ToTensor(),                    # 转成张量
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化，把像素值映射到 [-1,1]
    ])
    dataset = CIFAR10(root="./cifar10", train=True, transform=transform, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize student and teacher networks
output_dim = 128  # Example output dimension，特征向量维度
gs = ViT(output_dim).to(device)   # student 学生网络
gt = ViT(output_dim).to(device)   # teacher 教师网络
gt.load_state_dict(gs.state_dict())  # Teacher network initially copies the student weights（教师初始时拷贝学生权重）

# Initialize center
# 中心向量 C：用于 centering，初始为全 0，形状 (128)
C = torch.zeros(output_dim, device=device) # vit输出的特征是128，(128)

# DINO-specific loss function H
def H(t, s, C, tps, tpt):
    """DINO 的损失函数（交叉熵）：让学生输出 s 去逼近教师输出 t。
    参数：t=教师输出，s=学生输出，C=中心，tps=学生温度，tpt=教师温度。
    """
    t = t.detach()  # stop gradient for teacher，教师不回传梯度（只用它做监督信号）
    s = F.softmax(s / tps, dim=1) # 学生输出用较大温度 tps 做 softmax（分布更平滑），(B,128)
    t = F.softmax((t - C) / tpt, dim=1) # 教师输出先减中心 C 再用较小温度 tpt（分布更尖锐），(B,128)
    # 交叉熵：-Σ t*log(s)，越小说明学生越接近教师
    return - (t * torch.log(s)).sum(dim=1).mean()

# Mock function for augmentations (in a real scenario we would use more complex augmentations)
def augment(x):
    """模拟数据增强：实际 DINO 会做多视图复杂增强，这里简单地加一点高斯噪声。"""
    return x + 0.1 * torch.randn_like(x)

# Set up hyperparameters
batch_size = 64
tps = 0.1  # Temperature for student softmax，学生 softmax 温度
tpt = 0.07  # Temperature for teacher softmax，教师 softmax 温度（更小→输出更尖锐）
l = 0.6  # Network momentum rate，教师参数动量系数（越大教师更新越慢）
m = 0.5  # Center momentum rate，中心 C 的动量系数

# Optimizer for the student network
# 只优化学生网络 gs 的参数（教师不靠梯度更新）
optimizer = optim.SGD(gs.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4)

# Training loop
loader = get_dataloader(batch_size)
for x, _ in loader:  # We don't need labels for self-supervised learning（自监督，不用标签，用 _ 忽略）
    x = x.to(device) # (B,3,224,224)
    x1, x2 = augment(x), augment(x)  # random views，对同一张图做两次不同增强，得到两个视图
    s1, s2 = gs(x1), gs(x2)  # student output，学生处理两个视图，(B,128)
    t1, t2 = gt(x1), gt(x2)  # teacher output，教师处理两个视图，(B,128)
    # 交叉损失：用视图1的教师 t1 监督视图2的学生 s2，反之亦然；/2 取平均
    loss = H(t1, s2, C, tps, tpt)/2 + H(t2, s1, C, tps, tpt)/2 # divide by 2 for combined loss
    loss.backward()  # back-propagate，反向传播计算梯度
    optimizer.step()  # SGD update for student，更新学生参数
    optimizer.zero_grad()  # Clear gradients，清空梯度，为下一步做准备

    # Teacher and center updates
    # 教师网络和中心 C 的更新都不需要梯度
    with torch.no_grad():
        # 教师参数 = l*旧教师 + (1-l)*学生，即教师缓慢地向学生靠拢（指数移动平均）
        for teacher_param, student_param in zip(gt.parameters(), gs.parameters()):
            teacher_param.data = l * teacher_param.data + (1 - l) * student_param.data
        # 中心 C 也用动量更新，趋向于教师输出的均值
        C = m * C + (1 - m) * torch.cat([t1, t2]).mean(dim=0)

print("Training complete!")