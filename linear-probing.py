"""
Linear Probing（线性探测）与 Fine-Tuning（全量微调）的对比示例。

核心思想（小白版）：
    这两个类都用预训练好的 ViT 来做图像分类（CIFAR10 共 10 类），区别在于“训练哪些参数”：
      - FineTuning（微调）：整个 ViT 的参数都参与训练（更新所有权重），效果好但成本高、容易过拟合。
      - LinearProbing（线性探测）：冻结整个 ViT（不训练），只训练最后新加的一个线性分类头。
        常用于评估预训练特征的好坏：特征越好，只训一个线性层就能取得不错效果。
"""

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


class FineTuning(nn.Module):
    """全量微调：加载预训练 ViT，所有参数都可以被训练更新。"""
    def __init__(self, num_classes):
        super(FineTuning, self).__init__()
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True)  # 预训练 ViT
        # 把分类头换成输出 num_classes 维的新层（默认参数 requires_grad=True，整个模型都会训练）
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)


class LinearProbing(nn.Module):
    """线性探测：冻结预训练 ViT 的所有参数，只训练最后新加的分类头。"""
    def __init__(self, num_classes):
        super(LinearProbing, self).__init__()
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True)
        # 冻结主干网络：requires_grad=False 表示不参与梯度更新（不训练）
        for param in self.vit.parameters():
            param.requires_grad = False

        # Replace the classifier head
        # 重新创建分类头：新建的层默认可训练，所以只有这个头会被更新
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)


def load_cifar10_dataset():
    """加载 CIFAR10 数据集，返回 DataLoader（缩放到 224x224，batch_size=4）。"""
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_dataset = CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    return loader


def main():
    # 加载数据集
    dataset = load_cifar10_dataset()
    # 选择要用的模型：默认用线性探测；若想对比全量微调，换用下面的 FineTuning
    model = LinearProbing(num_classes=10)
    # model = FineTuning(num_classes=10)

    # 遍历每个 batch（这里只演示前向传播和损失计算）
    for images, labels in dataset:
        logits = model(images)  # 预测得分，(B,10)

        # classification loss（交叉熵分类损失）
        loss = torch.nn.CrossEntropyLoss()(logits, labels)

        # 输出损失
        print(loss)

if __name__ == "__main__":
    main()