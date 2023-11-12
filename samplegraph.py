# -*-coding: utf-8 -*-
# @Time    : 2023/8/30 17:01
# @Author  : Liangliang
# @File    : samplegraph.py
# @Software: PyCharm
import random

import dgl
import torch
import time


def sample_graph(i, g, arg):
    # 开始抽取子图
    start = time.time()
    g_sub, _ = dgl.khop_in_subgraph(g, i, k=arg.k_hop)
    num = -1
    while g_sub.number_of_nodes() > arg.subgraph_nodes_max_num or g_sub.number_of_edges() > arg.subgraph_edges_max_num\
            or g_sub.number_of_nodes() <= arg.subgraph_nodes_min_num or \
            g_sub.number_of_edges() <= arg.subgraph_edges_min_num:
        num += 1
        nodes = g_sub.nodes()
        # 防止程序长时间处于节点采样过程中
        if num < 2:
            degree = g_sub.in_degrees(nodes)
            degree = degree / degree.sum()
            degree = torch.sigmoid(degree)
            prob = torch.rand(degree.size())
            nodes = g_sub.ndata[dgl.NID][prob < degree]
            nodes = torch.cat((nodes, torch.tensor([i]))).unique()
            g_sub = dgl.node_subgraph(g, nodes.to(torch.int32))
        else:
            g_sub, _ = dgl.khop_in_subgraph(g, i, k=1)
            if g_sub.number_of_nodes() > arg.subgraph_nodes_max_num or \
                    g_sub.number_of_edges() > arg.subgraph_edges_max_num:
                # 获取后继节点
                successors = g.successors(i)[0:arg.subgraph_nodes_max_num]
                successors = torch.cat((successors, torch.tensor([i]))).unique()
                g_sub = dgl.node_subgraph(g, successors.to(torch.int32))
            break
    g_sub = dgl.add_self_loop(g_sub)
    g_sub = dgl.to_bidirected(g_sub, copy_ndata=True)
    g_sub = dgl.to_simple(g_sub)
    end = time.time()
    if random.uniform(0, 1) <= 0.001:
        print('sample graph: {} cost: {}s'.format(i, end - start))
    return g_sub

# g = dgl.graph(([0,1,0,2,1,3,3,2,3,4,5,4,4,6],[1,0,2,0,3,1,2,3,4,3,4,5,6,4]),num_nodes=7)


def sample_graph_k_hop(i, g, args):
    start = time.time()
    blocks = []
    # featout建议参数化
    featout = [100, 50]
    # 定义采样函数
    sampler = dgl.dataloading.MultiLayerNeighborSampler(featout, output_device=args.device)
    dataloader = dgl.dataloading.DataLoader(
        g,
        torch.tensor([i], dtype=torch.int32).to(args.device),
        sampler,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    for _, _, temp_blocks in dataloader:
        for block in temp_blocks:
            blocks += [block.to(args.device)]
    end = time.time()
    if random.uniform(0, 1) <= 0.001:
        print('sample graph: {} cost: {}s'.format(i, end - start))
    return blocks


