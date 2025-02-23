import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch_geometric.nn import SAGEConv, HypergraphConv

from models.hypergcn import get_HyperGCN_He_dict


class HyperTensorGraphConvolution(Module):
    def __init__(self, a, b):
        super(HyperTensorGraphConvolution, self).__init__()
        self.a, self.b = a, b
        self.W = Parameter(torch.FloatTensor(a, b))
        self.bias = Parameter(torch.FloatTensor(b))
        # self.edge_count = edge_count
        self.reset_parameters()

    def reset_parameters(self):
        std = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, structure, H, power, num_sample):
        # self.edge_count = self.edge_count.cuda()
        W, b = self.W, self.bias
        AH = signal_shift_hypergraph_sample(structure, H, power, num_sample)
        AHW = torch.mm(AH, W)
        output = AHW + b
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.a) + ' -> ' \
            + str(self.b) + ')'


class HyperSAGE(nn.Module):
    @staticmethod
    def generate_hyperedge_dict(data):
        data0 = data.clone().cpu()
        He_dict = get_HyperGCN_He_dict(data0)
        data.hyperedge_dict = He_dict
        for e, nodes in data.hyperedge_dict.items():
            data.hyperedge_dict[e] = torch.tensor(nodes)
        return data

    def __init__(self, num_features, num_classes, args):
        """
        d: initial node-feature dimension
        h: number of hidden units
        c: number of classes
        """
        super(HyperSAGE, self).__init__()
        d, l, c = num_features, args.All_num_layers, num_classes

        h = [d]
        for i in range(l - 1):
            power = l - i + 2
            h.append(2 ** power)
        h.append(c)

        if args.MLP_hidden >= 16:
            print('Caution: Too large hidden dimension. Running very slow.')

        self.hgc1 = HyperTensorGraphConvolution(d, 16)
        self.hgc2 = HyperTensorGraphConvolution(16, c)
        self.do, self.l = args.dropout, args.All_num_layers
        self.power = args.HyperSAGE_power
        self.num_sample = args.HyperSAGE_num_sample

    def reset_parameters(self):
        self.hgc1.reset_parameters()
        self.hgc2.reset_parameters()

    def forward(self, data):
        """
        an l-layer GCN
        """
        H = data.x
        structure = data.hyperedge_dict
        do, l = self.do, self.l
        power = self.power
        num_sample = self.num_sample
        H = F.relu(self.hgc1(structure, H, power, num_sample))
        H = F.dropout(H, do, training=self.training)
        H = self.hgc2(structure, H, power, num_sample)
        return F.log_softmax(H, dim=1)


class OGBHyperSAGE(torch.nn.Module):
    """
    No normalisation is needed according to
    https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/products/gnn.py
    """

    def __init__(self, info, *args, **kwargs):
        super(OGBHyperSAGE, self).__init__()

        dropout = 0.5  # info["dropout"]
        num_layers = 3  # info['num_layers']
        hidden_channels = 256  # info['hidden_channels']
        in_channels = info["num_node_features"]
        out_channels = info["num_classes"]

        self.convs = torch.nn.ModuleList()
        self.hyper_convs = torch.nn.ModuleList()

        self.hyper_convs.append(HypergraphConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        for _ in range(num_layers):
            self.hyper_convs.append(HypergraphConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.hyper_convs.append(HypergraphConv(hidden_channels, out_channels))

        self.lp = torch.nn.Linear(2 * info["num_classes"], info["num_classes"])

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data, *args, **kwargs):
        x, adj_t, hyperedge_index = data.x, data.edge_index, data.hyperedge_index
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_simple = self.convs[-1](x, adj_t)
        x = data.x
        for conv in self.hyper_convs[:-1]:
            x = conv(x, hyperedge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.hyper_convs[-1](x, hyperedge_index)
        x = self.lp(torch.cat([x_simple.unsqueeze(0), x.unsqueeze(0)], dim=2)).squeeze(0)
        return torch.log_softmax(x, dim=-1)


def signal_shift_graph2(hypergraph, edge_count, H):
    new_signal = H.clone()
    for edge, nodes in hypergraph.items():
        for node_i in nodes:
            neighbor_nodes = nodes[nodes != node_i]
            new_signal[node_i] = new_signal[node_i] + torch.sum(H[neighbor_nodes], dim=0)

    H = new_signal / (edge_count + 1)
    return H


def signal_shift_hypergraph2(hypergraph, H):
    # new_signal = torch.zeros(H.shape[0],H.shape[1]).cuda()
    new_signal = H.clone()
    for edge, nodes in hypergraph.items():
        for node_i in nodes:
            neighbor_nodes = (nodes[nodes != node_i]).to(dtype=torch.long)
            node_i = node_i.to(dtype=torch.long)
            new_signal[node_i] = new_signal[node_i] + torch.sum(H[neighbor_nodes], dim=0) / (len(nodes) - 1)
    return H


def signal_shift_hypergraph_power(hypergraph, H, power):
    min_value, max_value = 1e-7, 1e1
    torch.clamp_(H, min_value, max_value)
    # new_signal = torch.zeros(H.shape[0],H.shape[1]).cuda()
    new_signal = H.clone()
    for edge, nodes in hypergraph.items():
        for node_i in nodes:
            neighbor_nodes = (nodes[nodes != node_i]).to(dtype=torch.long)
            node_i = node_i.to(dtype=torch.long)
            new_signal[node_i] = new_signal[node_i] + torch.pow(
                torch.sum(torch.pow(H[neighbor_nodes], power), dim=0) / (len(nodes) - 1), 1 / power)

    return normalize(new_signal)


def signal_shift_hypergraph_sample(hypergraph, H, power, num_sample):
    min_value, max_value = 1e-7, 1e1
    torch.clamp_(H, min_value, max_value)
    # new_signal = torch.zeros(H.shape[0],H.shape[1]).cuda()
    new_signal = H.clone()

    for edge, nodes in hypergraph.items():
        for node_i in nodes:
            neighbor_nodes = (nodes[nodes != node_i]).to(dtype=torch.long)
            node_i = node_i.to(dtype=torch.long)
            if (len(neighbor_nodes) > num_sample):
                shuffled_neighborhood = H[neighbor_nodes][torch.randperm(H[neighbor_nodes].size()[0])]
                new_signal[node_i] = new_signal[node_i] + torch.pow(
                    torch.sum(torch.pow(shuffled_neighborhood[0:num_sample], power), dim=0) / (len(nodes) - 1),
                    1 / power)
            else:
                new_signal[node_i] = new_signal[node_i] + torch.pow(
                    torch.sum(torch.pow(H[neighbor_nodes], power), dim=0) / (len(nodes) - 1), 1 / power)
    return normalize(new_signal)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = torch.tensor(mx.sum(1))
    r_inv = torch.pow(rowsum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    mx = mx * r_inv[:, None]
    return mx


def normalize_np(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
