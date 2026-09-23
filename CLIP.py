"""
CLIP（Contrastive Language-Image Pre-training，对比式图文预训练）的最小实现示例。

核心思想（小白版）：
    训练两个编码器——一个负责“看图”（图像编码器 ViT），一个负责“读文”（文本编码器 BERT）。
    把一张图片和它对应的文字分别编码成向量后，希望“配对的图片-文字”向量尽量靠近（相似度高），
    “不配对的图片-文字”向量尽量远离。这样模型就学会了图像和文本在同一个语义空间里对齐。

    训练时用了一个巧妙技巧：一个 batch 里有 B 张图片、B 段文字，
    正确的配对只有对角线上的 B 组（第 i 张图配第 i 段文字），其余 B*B-B 组都是错误配对（负样本）。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from transformers import BertModel, BertTokenizer
import timm
import numpy as np

# 图像编码器 - 使用ViT
class ViT(nn.Module):
    """图像编码器：把一张图片 (B,3,224,224) 变成一个特征向量 (B,output_dim)。"""
    def __init__(self, output_dim):
        super(ViT, self).__init__()
        # 使用来自timm的ViT模型
        # pretrained=True 表示加载在 ImageNet 上预训练好的权重；num_classes=output_dim 决定输出向量维度
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=output_dim)

    def forward(self, x):
        # 直接调用 ViT，输入图像张量，输出图像特征向量
        return self.vit(x)

# 文本编码器 - 使用BERT
class TextEncoder(nn.Module):
    """文本编码器：把一段文字变成一个特征向量 (B,768)。"""
    def __init__(self):
        super(TextEncoder, self).__init__()

        # BERT 模型的本地路径（需要提前下载好放到该目录，否则会报错）
        BERT_LOCAL_PATH = './bert-base-uncased'
        self.model = BertModel.from_pretrained(BERT_LOCAL_PATH)       # BERT 主体网络
        self.tokenizer = BertTokenizer.from_pretrained(BERT_LOCAL_PATH)  # 分词器：把文字转成 BERT 能读懂的编号

    def forward(self, texts):
        # 文本通过BERT
        # tokenizer 把字符串列表转成张量：padding=True 补齐到同一长度，truncation=True 过长则截断
        encoded_input = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**encoded_input)
        # last_hidden_state 形状为 (B, 序列长度, 768)；取第 0 个 token（即 [CLS]）作为整句话的代表向量
        return outputs.last_hidden_state[:, 0, :]

# 加载CIFAR10数据集
def load_cifar10_dataset():
    """下载并加载 CIFAR10 数据集，返回数据加载器 loader 和类别名列表 classes。"""
    # 数据预处理：先把图片缩放到 224x224（ViT 要求的输入尺寸），再转成张量
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    # train=True 使用训练集；download=True 若本地没有则自动下载到 ./cifar10
    train_dataset = CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
    # DataLoader 负责按 batch_size 分批、打乱数据，方便训练
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    classes = train_dataset.classes  # CIFAR10 的 10 个类别名，如 ['airplane', 'automobile', ...]
    return loader, classes


class CLIP(nn.Module):
    """CLIP 主模型：把图像编码器和文本编码器组合起来，输出图文相似度矩阵。"""
    def __init__(self, image_output_dim, text_output_dim):
        super(CLIP, self).__init__()
        self.image_encoder = ViT(image_output_dim)
        self.text_encoder = TextEncoder()

        # 因为图像和文本emb可能维度不同(图像512，文本768)，所以需要对图像和文本的emb再经过一层以将维度持平
        self.W_i = nn.Parameter(torch.randn(image_output_dim, text_output_dim))
        self.W_t = nn.Parameter(torch.randn(768, text_output_dim))  # BERT-base的最后隐藏层大小为768

    def forward(self, images, texts):
        I_f = self.image_encoder(images) # 图像特征 (B,3,224,224) -> (B, 512)
        T_f = self.text_encoder(texts) # 文本特征 （B段文字）-> (B, 768)

        # 调整维度：用可学习的权重矩阵把图像、文本特征都映射到同一个维度(text_output_dim)
        I_e = torch.matmul(I_f, self.W_i) # (B, 512) @ (512, text_dim) -> (B, text_dim)
        T_e = torch.matmul(T_f, self.W_t) # (B, 768) @ (768, text_dim) -> (B, text_dim)

        # 计算相似度矩阵：I_e 与 T_e 转置相乘，得到每张图与每段文字的两两相似度
        # logits[i][j] 表示第 i 张图和第 j 段文字的匹配分数，对角线才是正确配对
        logits = torch.matmul(I_e, T_e.T) # (B,B)
        return logits


# 主函数
def main():
    # 加载数据集
    dataset, classes = load_cifar10_dataset()
    # 创建 CLIP 模型：图像和文本特征最终都统一到 512 维
    clip_model = CLIP(image_output_dim=512, text_output_dim=512)

    # 遍历每一个 batch（这里只是演示前向传播和损失计算，没有做真正的反向传播训练）
    for images, labels in dataset:
        # 获取一个小批量的图像和标签
        # 用标签对应的类别名当作这段图片的“文本描述”，例如 label=0 -> 'airplane'
        texts = [classes[label] for label in labels]

        logits = clip_model(images, texts) # 相似度矩阵 (B,B)
        # 正确答案：第 i 张图应配第 i 段文字，所以目标标签就是 [0,1,2,3]（对角线位置）
        labels = torch.arange(logits.shape[0]) # (0,1,2,3)

        # 计算损失（对称的交叉熵）：
        # loss_i：以“图像”为查询，让每张图在 B 段文字里选中正确的那一段（按行分类）
        # loss_t：以“文本”为查询，让每段文字在 B 张图里选中正确的那一张（按列分类，故用 logits.T）
        loss_i = torch.nn.CrossEntropyLoss()(logits, labels)
        loss_t = torch.nn.CrossEntropyLoss()(logits.T, labels)
        loss = (loss_i + loss_t) / 2  # 两个方向取平均，得到最终对比损失

        # 输出损失
        print(loss)

if __name__ == "__main__":
    main()
