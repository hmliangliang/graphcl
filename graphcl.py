# -*-coding: utf-8 -*-
# @Time    : 2023/8/30 10:13
# @Author  : Liangliang
# @File    : graphcl.py
# @Software: PyCharm
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv
from dgl.nn import GATv2Conv


class Graphcl(nn.Module):
    def __init__(self, in_feats, hiddern_feats, output_dim, dropoutrate=0.2, num_heads=2):
        super(Graphcl, self).__init__()
        self.dropout = dropoutrate
        self.mask_t = nn.Linear(in_feats, in_feats, bias=True)
        self.mask_s = nn.Linear(in_feats, in_feats, bias=True)
        self.layer1 = SAGEConv(in_feats, hiddern_feats, 'mean')
        self.layer2 = SAGEConv(hiddern_feats, output_dim, 'mean')
        self.layer21 = GATv2Conv(in_feats, hiddern_feats, num_heads)
        self.layer22 = GATv2Conv(hiddern_feats, output_dim, num_heads)
        self.layer3 = nn.Linear(output_dim + in_feats, output_dim, bias=True)

    def forward(self, blocks, feat, training=False):
        # 计算teacher网络的输出
        #进行mask操作
        if training:
            # 训练过程
            mask_t = self.mask_t(feat)
            mask_t = F.sigmoid(mask_t)
            h_t = feat * mask_t
            h_t = self.layer1(blocks, h_t)
            h_t = F.gelu(h_t)
            h_t = F.dropout(h_t, self.dropout, training=training)
            h_t = self.layer2(blocks, h_t)
            h_t = F.gelu(h_t)
            h_t = torch.concat([h_t, feat], dim=1)
        else:
            # 推理过程
            feat = blocks[0].srcdata["feat"]
            h_t = feat
            h_t = self.layer1(blocks[0], h_t)
            h_t = F.gelu(h_t)
            h_t = self.layer2(blocks[1], h_t)
            h_t = F.gelu(h_t)
            h_t = torch.concat([h_t, blocks[1].dstdata["feat"]], dim=1)
        h_t = self.layer3(h_t)
        h_t = F.leaky_relu(h_t)
        h_t = F.normalize(h_t, p=2, dim=1)

        # 计算student网络的输出
        if training:
            # 训练过程
            # 进行mask操作
            mask_s = self.mask_s(feat)
            mask_s = F.sigmoid(mask_s)
            h_s = feat * mask_s

            h_s = self.layer21(blocks, h_s)
            h_s = F.gelu(h_s)
            h_s = h_s.mean(dim=1)
            h_s = F.dropout(h_s, self.dropout, training=training)
            h_s = self.layer22(blocks, h_s)
        else:
            # 推理过程
            feat = blocks[0].srcdata["feat"]
            h_s = feat
            h_s = self.layer21(blocks[0], h_s)
            h_s = F.gelu(h_s)
            h_s = h_s.mean(dim=1)
            h_s = self.layer22(blocks[1], h_s)
        h_s = F.gelu(h_s)
        h_s = h_s.mean(dim=1)
        h_s = F.normalize(h_s, p=2, dim=1)
        return [h_t, h_s]