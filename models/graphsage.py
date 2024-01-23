import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class SAGENet(torch.nn.Module):
    def __init__(self, info, mixture_cls=None, load_config=None, *args, **kwargs):
        super(SAGENet, self).__init__()
        dim = 32
        self.is_regression = info["is_regression"]
        self.is_edge_pred = info["is_edge_pred"]
        if self.is_regression:
            self.conv2 = SAGEConv(dim, dim, normalize=False)
            self.head = nn.Linear(dim, 1)
        elif info["is_edge_pred"]:
            self.conv2 = SAGEConv(dim, dim)
        else:
            self.conv2 = SAGEConv(dim, info["num_classes"], normalize=False)

        self.conv1 = SAGEConv(info["num_node_features"], dim, normalize=False)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        if self.is_regression:
            self.head.reset_parameters()

    def forward(self, data, *args, **kwargs):
        x, edge_index = data.x, data.edge_index
        if self.is_edge_pred:
            edge_index = args[0][0]  # the message passing edge index
        x = F.relu(
            self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
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


class OGBSAGE(torch.nn.Module):
    """
    No normalisation is needed according to
    https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/products/gnn.py
    """

    def __init__(self, info, *args, **kwargs):
        super(OGBSAGE, self).__init__()

        dropout = 0.5  # info["dropout"]
        num_layers = 3  # info['num_layers']
        hidden_channels = 256  # info['hidden_channels']
        in_channels = info["num_node_features"]
        out_channels = info["num_classes"]

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data, *args, **kwargs):
        x, adj_t = data.x, data.edge_index
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)
