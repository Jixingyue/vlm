"""
Visual Prompt Tuning（视觉提示微调，VPT）的最小实现示例——属于“参数高效微调”方法。

核心思想（小白版）：
    借鉴 NLP 里的 prompt（提示）思想：冻结预训练 ViT 的原有参数，只往输入序列里加入
    少量可学习的“提示 token”（learnable tokens），只训练这些提示 token 和分类头。

    两种模式：
      - shallow（浅层）：只在输入层插入一次提示 token（拼在序列最前面）；
      - deep（深层）：每个 Transformer Block 都重新插入/替换提示 token，表达能力更强。
"""

import torch
import torch.nn as nn
import timm
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


class VisualPromptTuning(nn.Module):
    """视觉提示微调模型：冻结 ViT 主体，只训练可学习的提示 token 和分类头。

    参数：num_classes=分类数，hidden_dim=预留参数（本实现未使用），
    num_learnable_tokens=提示 token 的数量，mode='shallow'/'deep'。
    """
    def __init__(self, num_classes, hidden_dim, num_learnable_tokens=1, mode='shallow'):
        super(VisualPromptTuning, self).__init__()
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=num_classes)
        self.num_learnable_tokens = num_learnable_tokens
        self.mode = mode

        # Replace classifier head (just to ensure it is trainable)
        # 重建分类头，确保它是可训练的
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

        # Create learnable tokens for shallow mode
        # 浅层模式用的可学习提示 token，形状 (1,10,384)，embed_dim=384
        self.learnable_tokens = nn.Parameter(torch.randn(1, num_learnable_tokens, self.vit.embed_dim))  # (1,10,384)

        if mode == 'deep':
            # For deep mode, create learnable tokens for each block
            # 深层模式：为每个 Block 都准备一组可学习提示 token
            self.deep_learnable_tokens = nn.ParameterList([
                nn.Parameter(torch.randn(1, num_learnable_tokens, self.vit.embed_dim)) # (1,10,384)
                for _ in self.vit.blocks
            ])

        # Freeze all but the head and prompt tokens
        # 先冻结 ViT 的所有参数（注意：self.learnable_tokens / deep_learnable_tokens 不属于 self.vit，不受影响）
        for param in self.vit.parameters():
            param.requires_grad = False
        # 再把分类头解冻（可训练）；提示 token 本身作为 nn.Parameter 默认可训练
        for param in self.vit.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        # 手动拆解 ViT 前向过程，以便插入提示 token
        x = self.vit.patch_embed(x)  # 切 patch 并嵌入，(B,196,384)

        if self.vit.cls_token is not None:
            cls_tokens = self.vit.cls_token.expand(x.size(0), -1, -1)  # [CLS] token，(B,1,384)
            x = torch.cat((cls_tokens, x), dim=1) # 拼到最前面，(B,197,384)
        if self.vit.pos_embed is not None:
            x = x + self.vit.pos_embed  # 加位置编码
        x = self.vit.pos_drop(x)

        if self.mode == 'shallow':
            # concatenate the learnable tokens at the beginning, deep在后面会做
            # 浅层：把提示 token 拼到序列最前面；expand:(1,10,384)->(B,10,384), cat:(B,197,384)->(B,207,384)
            x = torch.cat((self.learnable_tokens.expand(x.size(0), -1, -1), x), dim=1) # expand: (1,10,384)->(B,10,384), cat:(B,197,384)->(B,207,384)

        for i, block in enumerate(self.vit.blocks):
            if self.mode == 'deep':
                # For deep mode, replace the learnable tokens at each block
                # 深层：在每个 Block 前，用本层专属的提示 token 覆写序列中对应位置（本实现简化处理）
                x[:, 1:self.num_learnable_tokens + 1] = self.deep_learnable_tokens[i] #(1,10,384)
            x = block(x)

        x = self.vit.norm(x) # 最后归一化，(B,197,384)
        if self.vit.cls_token is not None:
            x = x[:, 0]  # 取 [CLS] 位置作为整图表示

        return self.vit.head(x)  # 分类头输出 logits，(B,num_classes)


def load_cifar10_dataset():
    """加载 CIFAR10 数据集，返回 DataLoader（缩放到 224x224，batch_size=4）。"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    return loader


def main():
    # Load the CIFAR-10 dataset
    dataset = load_cifar10_dataset()

    # Initialize the model in 'deep' mode with a single learnable token
    # 创建模型：这里用 deep 模式，10 个可学习提示 token
    model = VisualPromptTuning(num_classes=10, hidden_dim=64, num_learnable_tokens=10, mode='deep')

    # 遍历每个 batch（这里只演示前向传播和损失计算）
    for images, labels in dataset:
        logits = model(images)  # 预测得分，(B,10)

        loss = torch.nn.CrossEntropyLoss()(logits, labels)  # 交叉熵分类损失

        # Output the losses
        print(f'Loss: {loss.item()}')


if __name__ == "__main__":
    main()
