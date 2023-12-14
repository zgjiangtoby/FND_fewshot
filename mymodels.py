import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter_Origin(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(Adapter_Origin, self).__init__()
        self.fc = torch.nn.Linear(512*2, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, feature_dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.cross = nn.Linear(feature_dim, 2)
    def forward(self, txt_feat, img_feat):
        # 生成查询、键和值
        query = self.query(txt_feat)
        key = self.key(img_feat)
        value = self.value(img_feat)

        # 计算注意力分数
        attn_scores = torch.matmul(query, key.transpose(-2, -1))
        attn_scores = F.softmax(attn_scores, dim=-1)

        # 应用注意力分数
        attn_output = torch.matmul(attn_scores, value)
        attn_output = self.cross(attn_output)

        return attn_output

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # 计算两个输出之间的欧几里得距离
        euclidean_distance = F.pairwise_distance(output1, output2)

        # 计算对比损失
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

class Adapter_V1(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(Adapter_V1, self).__init__()
        self.fc = nn.Linear(512 * 2, num_classes)
        self.fc_txt = nn.Linear(512, num_classes)
        self.fc_img = nn.Linear(512, num_classes)
        self.cross_attention = CrossAttention(512)
        self.fc_meta = nn.Linear(num_classes * 4, num_classes)


    def forward(self, txt, img, fused):
        # 获取文本和图像的输出
        txt_out = self.fc_txt(txt)
        img_out = self.fc_img(img)
        fused_out = self.fc(fused)
        # 应用交叉注意力机制
        ti_attn_out = self.cross_attention(txt, img)
        it_attn_out = self.cross_attention(img, txt)

        attn_out = ti_attn_out + it_attn_out

        # 合并来自基学习器的输出
        combined_out = torch.cat((txt_out, img_out, fused_out, attn_out), dim=1)

        meta_out = self.fc_meta(combined_out)

        return txt_out, img_out, meta_out


