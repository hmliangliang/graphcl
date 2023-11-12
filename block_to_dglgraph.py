# -*-coding: utf-8 -*-
# @Time    : 2023/11/10 11:24
# @Author  : Liangliang
# @File    : block_to_dglgraph.py
# @Software: PyCharm
import dgl


# 该函数的功能是把block graph合并成一个大的DGLGraph
def block_to_graph(blocks):
    # 从 dgl.DGLBlock 对象创建一个 dgl.DGLGraph 对象
    # graph = []
    # n = len(blocks)
    # for i in range(n):
    #     block = blocks[i]
    #     nodes_list = list(set(block.srcnodes().tolist() + block.dstnodes().tolist()))
    #     g = dgl.graph(block.edges(), num_nodes=len(nodes_list))
    #     keys = set(block.srcdata.keys()) | set(block.dstdata.keys())
    #     for key in keys:
    #         if key in ["feat", "label"]:
    #             g.ndata[key] = block.ndata[key]["_N"]
    #     graph.append(g)
    # 合并各个子图
    graph = dgl.merge(blocks)
    graph = graph.to("cpu")
    graph = dgl.add_self_loop(graph)
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.to_simple(graph)
    return graph