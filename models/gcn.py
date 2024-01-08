import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNNet(torch.nn.Module):
    def __init__(self, info, *args, **kwargs):
        super().__init__()
        dim = 32
        self.conv1 = GCNConv(info["num_node_features"], dim)
        self.is_regression = info["is_regression"]
        self.is_edge_pred = info["is_edge_pred"]
        if info["is_regression"]:
            self.conv2 = GCNConv(dim, dim)
            self.head = nn.Linear(dim, 1)
        elif info["is_edge_pred"]:
            self.conv2 = GCNConv(dim, dim)
        else:
            self.conv2 = GCNConv(dim, info["num_classes"])

    def forward(self, data, *args, **kargs):
        x, edge_index = data.x, data.edge_index
        if self.is_edge_pred:
            edge_index = args[0][0]  # the message passing edge index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        if self.is_regression:
            x = self.head(x).squeeze()
        elif self.is_edge_pred:
            # args[1] is the edge_label_index
            edge_label_index = args[0][1]
            x = x[edge_label_index[0]] * x[edge_label_index[1]]
            x = torch.sum(x, -1)
        else:
            x = F.log_softmax(x, dim=1)
        return x


class OGBGCN(torch.nn.Module):
    def __init__(self, info, *args, **kwargs):
        """
        This model is used to replicate the performance of ogb
        GCN require the following normalization for improve the performance
            # Pre-compute GCN normalization.
            adj_t = data.adj_t.set_diag()
            deg = adj_t.sum(dim=1).to(torch.float)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
            data.adj_t = adj_t
        """
        super(OGBGCN, self).__init__()

        self.dropout = 0.5  # info["dropout"]
        num_layers = 3  # info['num_layers']
        hidden_channels = 256  # info['hidden_channels']
        in_channels = info["num_node_features"]
        out_channels = info["num_classes"]
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data, *args, **kargs):
        x, adj_t = data.x, data.adj_t
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)
