# -*-coding: utf-8 -*-
# @Time    : 2023/8/30 15:31
# @Author  : Liangliang
# @File    : execution.py
# @Software: PyCharm

import os
os.system("pip install dgl")
os.environ['DGLBACKEND'] = "pytorch"
os.system("pip install pandas")
import time
import datetime
import math
import random
import torch
import pandas as pd
import numpy as np
import dgl
import graphcl
import samplegraph
import block_to_dglgraph

# 设置随机种子点
random.seed(921208)
np.random.seed(921208)
torch.manual_seed(921208)
os.environ['PYTHONHASHSEED'] = "921208"


# 读取输入的图数据
def read_graph(args):
    '''读取图数据部分
    二维数组，每一个元素为边id，一行构成一条边
    '''
    path = args.data_input.split(',')[0]
    input_files = sorted([file for file in os.listdir(path) if file.find("part-") != -1])
    count = 0
    print("开始读取数据! {}".format(datetime.datetime.now()))
    data = pd.DataFrame()
    for file in input_files:
        count += 1
        print("当前正在处理第{}个文件,文件路径:{}......".format(count, os.path.join(path, file)))
        # 读取边结构数据
        data = pd.concat([data, pd.read_csv(os.path.join(path, file), sep=',', header=None)], axis=0)
    print("开始读取属性特征信息! {}".format(datetime.datetime.now()))
    # 读取属性特征信息
    path = args.data_input.split(',')[1]
    input_files = sorted([file for file in os.listdir(path) if file.find("part-") != -1])
    data_attr = pd.DataFrame()
    for file in input_files:
        # 读取属性特征数据
        data_attr = pd.concat([data_attr, pd.read_csv(os.path.join(path, file), sep=',', header=None)], axis=0)
    # 读取节点的属性特征数据
    data_attr = torch.from_numpy(data_attr.values.astype(np.float32))
    # 定义图结构
    g = dgl.graph((data.iloc[:, 0].to_list(), data.iloc[:, 1].to_list()), num_nodes=data_attr.shape[0],
                  idtype=torch.int32)
    # 转化为无向图
    g = dgl.to_bidirected(g)
    g.ndata["feat"] = data_attr
    g = dgl.add_self_loop(g)
    g = dgl.to_bidirected(g, copy_ndata=True)
    g = dgl.to_simple(g)
    return g


def loss_function(h_t, h_s, adj, args):
    adj = adj.to_dense()
    n0, m0 = h_t.shape
    n = torch.tensor([n0]).to(args.device)
    m = torch.tensor([m0]).to(args.device)
    neg_sample_num = torch.tensor([args.neg_sample_num]).to(args.device)
    lamda = torch.tensor([args.lamda]).to(args.device)
    q = torch.tensor([args.q]).to(args.device)
    # 链路预测loss
    loss_link = 1 / (n * m) * (torch.norm(h_t @ h_t.transpose(0, 1) - adj, p=1) + torch.norm(h_s @ h_s.transpose(0, 1)
                                                                                             - adj, p=1))
    del adj
    # 正样本对比学习loss
    loss_pos = 1 / n * torch.sum(torch.sum(h_t * h_s, dim=1) / (torch.norm(h_t, 2, dim=1) * torch.norm(h_s, 2, dim=1)))
    sample_matrix = torch.concat((h_t, h_s), dim=0)
    N = sample_matrix.shape[0]
    sample_matrix_t = sample_matrix[torch.randint(0, N, (1, n0 * args.neg_sample_num)), :][0]
    h_t = torch.reshape(h_t.repeat(1, args.neg_sample_num), [-1, m0])
    loss_neg1 = 1 / (n * neg_sample_num) * torch.sum(torch.sum(h_t * sample_matrix_t, dim=1) /
                                                      (torch.norm(h_t, 2, dim=1) * torch.norm(sample_matrix_t, 2,
                                                                                              dim=1)))

    h_s = torch.reshape(h_s.repeat(1, args.neg_sample_num), [-1, m0])
    loss_neg2 = 1 / (n * neg_sample_num) * torch.sum(torch.sum(h_s * sample_matrix_t, dim=1) /
                                                      (torch.norm(h_s, 2, dim=1) * torch.norm(sample_matrix_t, 2,
                                                                                              dim=1)))
    loss = loss_pos + lamda * loss_link + q * (loss_neg1 + loss_neg2) / 2
    return loss


def train(args):
    # 第一步读取图数据
    g = read_graph(args)
    g = g.to(args.device)
    args.input_dim = g.ndata["feat"].shape[1]
    print("开始执行训练过程! {}".format(datetime.datetime.now()))
    if args.env == "train":
        # 判断需要定义哪个模型
        if args.model_name == "graphcl":
            model = graphcl.Graphcl(args.input_dim, args.feat_dim, args.output_dim,
                                    dropoutrate=args.dropout, num_heads=args.num_heads)
            model = model.to(args.device)
    else:
        # train_incremental环境, 装载训练好的模型
        if args.model_name == "graphcl":
            model = graphcl.Graphcl(args.input_dim, args.feat_dim, args.output_dim,
                                    dropoutrate=args.dropout, num_heads=args.num_heads)
            cmd = "s3cmd get -r  " + args.model_output + "graphcl"
            os.system(cmd)
            checkpoint_path = "./graphcl/graphcl.pth"
            model.load_state_dict(torch.load(checkpoint_path))
            model = model.to(args.device)
            print("Model is loaded!")
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    beforeLoss = torch.tensor([2 ** 23]).to(args.device)
    stopNum = 0
    n = g.number_of_nodes()
    for epoch in range(args.epoch):
        print("开始第{}epoch的训练. {}".format(epoch, datetime.datetime.now()))
        loss = 0
        for sample_num in range(args.sample_num):
            # 随机采样一个节点
            i = random.randint(0, n - 1)
            blocks = samplegraph.sample_graph_k_hop(i, g, args)
            g_sub = block_to_dglgraph.block_to_graph(blocks)
            g_sub = g_sub.to(args.device)
            h_t, h_s = model(g_sub, g_sub.ndata["feat"], training=True)
            loss = loss_function(h_t, h_s, g_sub.adjacency_matrix(), args)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if sample_num % args.batch_num == 0:
                print("epoch:{}  sample_num:{}  loss:{} {}".format(epoch, sample_num, loss, datetime.datetime.now()))
        if loss < beforeLoss:
            beforeLoss = loss
            stopNum = 0
            # 保存模型
            os.makedirs("./graphcl", exist_ok=True)
            model = model.to(args.device)
            torch.save(model.state_dict(), './graphcl/graphcl.pth')
            cmd = "s3cmd put -r ./graphcl " + args.model_output
            os.system(cmd)
            print("在训练第{}个子图后,model模型已保存! {}".format(sample_num, datetime.datetime.now()))
        else:
            stopNum += 1
            if stopNum > args.stop_num:
                print("Early stop!")
                break


# 判断是否是二维列表
def is_list(lst):
    if isinstance(lst, list) and len(lst) > 0:
        return True
    else:
        return False


# 把推理结果写入文件系统中
def write(data, count, args):
    # 注意在此业务中data是一个二维list
    # 数据的数量
    print("开始写入第{}个文件数据. {}".format(count, datetime.datetime.now()))
    n = len(data)
    flag = is_list(data[0])
    if n > 0:
        start = time.time()
        with open(os.path.join(args.data_output, 'pred_{}.csv'.format(count)), mode="a") as resultfile:
            if flag:
                for i in range(n):
                    line = ",".join(map(str, data[i])) + "\n"
                    resultfile.write(line)
            else:
                line = ",".join(map(str, data)) + "\n"
                resultfile.write(line)
        cost = time.time() - start
        print("第{}个大数据文件已经写入完成,写入数据的行数{} 耗时:{}  {}".format(count, n, cost, datetime.datetime.now()))


# 单步推理过程
def inference_step(g, nodes, model, i, args):
    # 每个节点进行采样子图
    n = len(nodes)
    data = np.zeros((n, args.output_dim + 1)).astype(str)
    id = np.zeros((n, 1)).astype(str)
    temp_data = torch.zeros((n, args.output_dim)).cuda()
    s = -1
    for node in nodes:
        s += 1
        g_sub = samplegraph.sample_graph_k_hop(node, g, args)
        feat, _ = model(g_sub, "", training=False)
        id[s, 0] = str(node)
        temp_data[s, :] = feat[0, :]
    print("完成第{}个文件的子图推理过程! {}".format(i, datetime.datetime.now()))
    data[:, 0] = id[:, 0]
    data[:, 1:] = temp_data.to("cpu").detach().numpy().astype(str)
    write(data.tolist(), i, args)

    
# 执行推理过程
def inference(args):
    # 第一步读取图数据
    g = read_graph(args)
    g = g.to(args.device)
    args.input_dim = g.ndata["feat"].shape[1]
    # 第二步加载模型
    model = graphcl.Graphcl(args.input_dim, args.feat_dim, args.output_dim, args.dropout, num_heads=args.num_heads)
    cmd = "s3cmd get -r  " + args.model_output + "graphcl"
    os.system(cmd)
    checkpoint_path = "./graphcl/graphcl.pth"
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(args.device)
    print("Graph model is loaded!")
    # 第三步执行推理
    n = g.number_of_nodes()
    n_files = math.ceil(n / args.file_nodes_max_num)
    for i in range(n_files):
        nodes = [j for j in range(i * args.file_nodes_max_num, min((i + 1) * args.file_nodes_max_num, n))]
        print("一共{}个任务开始分发第{}个推理子任务 {}".format(n_files, i, datetime.datetime.now()))
        inference_step(g, nodes, model, i, args)

