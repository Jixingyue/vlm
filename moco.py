"""
MoCo（Momentum Contrast，动量对比学习）的最小实现示例。

核心思想（小白版）：
    这也是一种自监督对比学习。对同一张图片做两次增强，分别过两个编码器：
      - 查询编码器 f_q（query）：正常用梯度训练；
      - 键编码器 f_k（key）：用动量方式缓慢拷贝 f_q 的参数（保证键的稳定性）。
    同一张图的两个视图互为“正样本”（应该相似），而与队列 queue 里大量历史特征互为“负样本”（应该不相似）。

    MoCo 的两个关键点：
      1) 动量更新（momentum）：f_k 参数 = m*f_k + (1-m)*f_q，让键编码器变化缓慢，保证队列一致；
      2) 队列（queue）：维护一个很大的负样本队列（K=4096），无需超大 batch 也能有大量负样本。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

# 数据加载
transform = ToTensor()  # 把图片转成张量（像素值缩放到 [0,1]）
dataset = CIFAR10(root="./cifar10", train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=64, shuffle=True)


# 使用ResNet50
def get_resnet50(output_dim):
    """构造一个 ResNet50 编码器，并把最后的全连接层换成输出 output_dim 维特征。"""
    model = resnet50(pretrained=False)  # pretrained=False：自监督从头训练
    model.fc = nn.Linear(model.fc.in_features, output_dim)  # 替换分类头，输出维度改为 output_dim
    return model


# InfoNCE Loss
def info_nce_loss(q, k, queue, temperature=0.07):
    """InfoNCE 对比损失：让 q 与正样本 k 相似，与队列 queue 里的负样本不相似。
    （这里用到模块级全局变量 N=batch大小、C=特征维度，它们在下面定义）
    """
    # L2 归一化：把向量长度归一，使后面点积等价于余弦相似度
    q = nn.functional.normalize(q, dim=1, p=2)
    k = nn.functional.normalize(k, dim=1, p=2)
    queue = nn.functional.normalize(queue, dim=0, p=2)

    # 正样本相似度：每个 q 与其对应的 k 做点积，形状 (N,1)
    positive_similarity = torch.bmm(q.view(N,1,C), k.view(N,C,1))

    # 负样本相似度：每个 q 与队列里所有负样本做点积，形状 (N,K)
    negative_similarity = torch.mm(q, queue)

    # 拼接：第 0 列是正样本，后面是负样本，(N, 1+K)
    logits = torch.cat([positive_similarity.squeeze(-1), negative_similarity], dim=-1)

    # 正确答案总是第 0 列（正样本），所以标签全为 0
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)

    # 除以温度 temperature 后用交叉熵：温度越小，对困难负样本惩罚越大
    loss = nn.CrossEntropyLoss()(logits / temperature, labels)
    return loss



# 参数设定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

C = 1024          # 特征维度（编码器输出向量的长度）
N = loader.batch_size  # batch 大小（=64）
K = 4096          # 负样本队列长度
f_q = get_resnet50(C).to(device)  # 查询编码器
f_k = get_resnet50(C).to(device)  # 键编码器
f_k.load_state_dict(f_q.state_dict())  # 键编码器初始拷贝查询编码器的权重

m = 0.99  # 动量系数（越大键编码器更新越慢，越稳定）
queue = torch.randn(C, K).to(device)  # 初始化负样本队列，形状 (C,K)
queue_ptr = 0  # 队列写入指针（循环使用）

# 只优化查询编码器 f_q 的参数（f_k 靠动量更新，不靠梯度）
optimizer = optim.Adam(f_q.parameters(), lr=0.001)



# 模拟数据增强函数
def aug(x):
    """模拟数据增强：这里简单地加一点高斯噪声（实际 MoCo 会用更复杂的增强）。"""
    return x + 0.1 * torch.randn_like(x)


# 主循环
for x, _ in loader:  # 自监督学习不用标签，用 _ 忽略
    x = x.to(device)
    x_q = aug(x)  # 查询视图
    x_k = aug(x)  # 键视图

    q = f_q(x_q)  # 查询特征，(N,C)
    k = f_k(x_k)  # 键特征，(N,C)

    k = k.detach()  # 键不回传梯度（键编码器不靠梯度更新）

    loss = info_nce_loss(q, k, queue)  # 计算 InfoNCE 对比损失

    optimizer.zero_grad()  # 清空上一步梯度
    loss.backward()        # 反向传播
    optimizer.step()       # 更新查询编码器 f_q

    with torch.no_grad():
        # 动量更新键编码器：f_k = m*f_k + (1-m)*f_q
        for param_q, param_k in zip(f_q.parameters(), f_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

        # Update the keys queue
        # 把当前 batch 的键特征写入队列（循环覆写），作为后续 batch 的负样本
        batch_size = k.size(0)
        queue[:, queue_ptr:queue_ptr + batch_size] = k.T
        queue_ptr = (queue_ptr + batch_size) % K  # 指针循环前移

print("Training complete!")
