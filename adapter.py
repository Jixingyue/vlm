"""
Adapter（适配器微调）的最小实现示例——属于“参数高效微调”方法。

核心思想（小白版）：
    当我们有一个已经训练好的大模型（这里是预训练 ViT）时，不想把所有参数都重新训练（太贵）。
    Adapter 的做法是：冻结（freeze）原模型的所有参数，只在每个 Transformer Block 里插入一个
    很小的可训练模块（AdaptMLP），只训练这个小模块 + 新的分类头。这样只需训练极少量参数就能适配新任务。

    AdaptMLP 是一个“先降维再升维”的瓶颈结构：input_dim -> hidden_dim -> input_dim。
"""

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


class AdaptMLP(nn.Module):
    """适配器模块：一个瓶颈结构的小型 MLP（降维 -> ReLU -> 升维），参数量很少。"""
    def __init__(self, input_dim, hidden_dim):
        super(AdaptMLP, self).__init__()
        self.down_proj = nn.Linear(input_dim, hidden_dim)  # 降维：input_dim -> hidden_dim
        self.relu = nn.ReLU(inplace=True)
        self.up_proj = nn.Linear(hidden_dim, input_dim)    # 升维：hidden_dim -> input_dim（与原维度对齐）

    def forward(self, x):
        down = self.down_proj(x)
        relu = self.relu(down)
        up = self.up_proj(relu)
        return up

class Adapter(nn.Module):
    """带 Adapter 的 ViT：冻结原 ViT，只在每个 Block 里并行插入一个可训练的 AdaptMLP。"""
    def __init__(self, num_classes, hidden_dim):
        super(Adapter, self).__init__()
        # Load a pre-trained ViT model（加载预训练 ViT）
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True)

        # Freeze the rest of the parameters in the ViT model
        # 冻结原 ViT 的所有参数：requires_grad=False 表示这些参数不参与梯度更新
        for param in self.vit.parameters():
            param.requires_grad = False

        # add AdaptMLP layers in each block
        # ❗注意（小白提醒）：这里用的是普通 Python list 而不是 nn.ModuleList，
        #   会导致这些 AdaptMLP 的参数不被 PyTorch 自动注册（optimizer 可能拿不到它们）。
        #   若要真正训练，建议改成 self.adapt_mlps = nn.ModuleList()。
        self.adapt_mlps = []
        for i, block in enumerate(self.vit.blocks): #每个Block包括self attention layernorm以及这个mlp
            # Freeze all parameters except for those in AdaptMLP
            for param in block.parameters():
                param.requires_grad = False

            # 为每个 Block 创建一个 AdaptMLP，hidden_dim=64（输入维度取自原 Block 的 mlp.fc1）
            self.adapt_mlps.append(AdaptMLP(input_dim=self.vit.blocks[i].mlp.fc1.in_features, hidden_dim=hidden_dim)) # hidden_dim=64

        # Replace the classifier head with a new trainable layer
        # 把分类头换成新的、可训练的层（适配 num_classes 个类别）
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        # 下面手动拆解 ViT 的前向过程，目的是能在每个 Block 里插入 Adapter
        x = self.vit.patch_embed(x)  # Apply patch embedding, 切 patch 并嵌入，(B,196,384)
        if self.vit.cls_token is not None:
            cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)  # (1,1,384) -> (B,1,384)，分类用的 [CLS] token
            x = torch.cat((cls_token, x), dim=1)  # 拼到序列最前面，(B,197,384)
        if self.vit.pos_embed is not None:
            x = x + self.vit.pos_embed.expand(x.shape[0], -1, -1)  # 加上位置编码，(1,197,384) -> (B,197,384)
        x = self.vit.pos_drop(x)  # Apply dropout if present，位置 dropout

        # Pass the input through the modified ViT model
        for i, block in enumerate(self.vit.blocks):
            # Apply the original block's layer normalization and attention
            block_input = x  # 保存输入，用于残差连接

            # --- 子层 1：自注意力 + 残差 ---
            x = block.norm1(x) # (B,197,384)
            x = block.attn(x)  # 自注意力
            x = block_input + x  # 残差相加

            # Save the output of the original MLP
            # --- 子层 2：原 MLP（冻结） ---
            original_mlp_output = block.norm2(x)
            original_mlp_output = block.mlp(original_mlp_output) # (B,197,384)

            # Pass the same input through the AdaptMLP
            # --- 并行的 Adapter（可训练） ---
            adapt_mlp_output = self.adapt_mlps[i](x)  # (B,197,384)

            # Combine the outputs of the original MLP and AdaptMLP
            # 把原 MLP 输出与 Adapter 输出相加（Adapter 只学一个小的“修正量”）
            x = original_mlp_output + adapt_mlp_output

            # Finally, apply the head to get the classification output

        # Apply the final normalization and classification head
        x = self.vit.norm(x)  # 最后的归一化
        if self.vit.cls_token is not None:
            x = x[:, 0] # 取 [CLS] token 作为整图表示，(B,197,384) -> (B,384)
        return self.vit.head(x)  # 分类头输出 logits，(B,num_classes)



def load_cifar10_dataset():
    """加载 CIFAR10 数据集，返回 DataLoader（缩放到 224x224，batch_size=4）。"""
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_dataset = CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    return loader


def main():
    # 加载数据集
    dataset = load_cifar10_dataset()
    # 创建 Adapter 模型：10 分类，Adapter 隐藏维 64
    model = Adapter(num_classes=10, hidden_dim=64)

    # 遍历每个 batch（这里只演示前向传播和损失计算）
    for images, labels in dataset:
        logits = model(images)  # 预测得分，(B,10)

        # classification loss（交叉熵分类损失）
        loss = torch.nn.CrossEntropyLoss()(logits, labels)

        # 输出损失
        print(loss)

if __name__ == "__main__":
    main()