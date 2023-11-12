# -*-coding: utf-8 -*-
# @Time    : 2023/8/30 14:34
# @Author  : Liangliang
# @File    : main.py
# @Software: PyCharm


import argparse
import time
import execution
import torch



if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description='算法的参数')
    parser.add_argument("--env", help="运行的环境(train or inference)", type=str, default='train_incremental')
    parser.add_argument("--model_name", help="模型名称", type=str, default="graphcl")
    parser.add_argument("--device", help="学习率", type=str, default='cpu')
    parser.add_argument("--epoch", help="epoch的数目", type=int, default=20)
    parser.add_argument("--lr", help="学习率", type=float, default=0.00001)
    parser.add_argument("--dropout", help="dropout比率", type=float, default=0.15)
    parser.add_argument("--stop_num", help="执行Early Stopping的最低epoch", type=int, default=5)
    parser.add_argument("--neg_sample_num", help="采样子图的节点数目", type=int, default=3)
    parser.add_argument("--num_heads", help="多头自注意力机制的头数", type=int, default=8)
    parser.add_argument("--sample_num", help="采样子图的子图数目", type=int, default=1000)
    parser.add_argument("--batch_num", help="打印loss函数的周期", type=int, default=100)
    parser.add_argument("--input_dim", help="输入特征的维度", type=int, default=256)
    parser.add_argument("--feat_dim", help="隐含层神经元的数目", type=int, default=100)
    parser.add_argument("--output_dim", help="输出特征的维度", type=int, default=64)
    parser.add_argument("--subgraph_nodes_max_num", help="采样子图最大的节点数目", type=int, default=30)
    parser.add_argument("--subgraph_edges_max_num", help="采样子图最大的边数目", type=int, default=1500)
    parser.add_argument("--subgraph_nodes_min_num", help="采样子图最小的节点数目", type=int, default=5)
    parser.add_argument("--subgraph_edges_min_num", help="采样子图最小的边数目", type=int, default=10)
    parser.add_argument("--k_hop", help="采样子图的跳连数目", type=int, default=1)
    parser.add_argument("--lamda", help="loss中的lamda参数", type=float, default=1)
    parser.add_argument("--q", help="loss中的q参数", type=float, default=1)
    parser.add_argument("--file_nodes_max_num", help="单个csv文件中写入数据的最大行数", type=int, default=10000)
    parser.add_argument("--data_input", help="输入数据的位置", type=str, default='')
    parser.add_argument("--data_output", help="数据的输出位置", type=str, default='')
    parser.add_argument("--model_output", help="模型的输出位置",
                        type=str, default='s3://general__lingqu/xxx/models/graphcl/')
    parser.add_argument("--tb_log_dir", help="日志位置", type=str, default='')
    args = parser.parse_args()
    if torch.cuda.is_available():
        args.device = "cuda"
    else:
        args.device = "cpu"
    if args.env == "train" or args.env == "train_incremental":
        execution.train(args)
    elif args.env == "inference":
        execution.inference(args)
    else:
        raise TypeError("args.env必需是train或train_incremental或inference！")
    end_time = time.time()
    print("算法总共耗时:{}".format(end_time - start_time))
