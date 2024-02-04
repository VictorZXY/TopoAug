import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, HypergraphConv

from models.edgnn import EquivSetGNN


class LPGCNHyperConvAblation(torch.nn.Module):
    def __init__(self, info, *args, **kwargs):
        super().__init__()
        dim = 64
        self.hyper1 = HypergraphConv(info["num_node_features"], dim)
        self.hyper2 = HypergraphConv(dim, dim)
        self.is_regression = info["is_regression"]
        if info["is_regression"]:
            self.conv1 = GCNConv(info["num_node_features"] + dim, dim)
            self.conv2 = GCNConv(dim, dim)
            self.lp = nn.Linear(dim, dim)
            self.head = nn.Linear(dim, 1)
        else:
            self.conv1 = GCNConv(info["num_node_features"] + dim, dim)
            self.conv2 = GCNConv(dim, info["num_classes"])
            self.lp = nn.Linear(info["num_classes"], info["num_classes"])

    def reset_parameters(self):
        self.hyper1.reset_parameters()
        self.hyper2.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lp.reset_parameters()
        if self.is_regression:
            self.head.reset_parameters()

    def forward(self, data, *args, **kwargs):
        x, edge_index, hyperedge_index = data.x, data.edge_index, data.hyperedge_index
        x_hyper = F.relu(self.hyper1(x, hyperedge_index))
        x_hyper = F.dropout(x_hyper, training=self.training)
        x_hyper = self.hyper2(x_hyper, hyperedge_index)
        x = torch.cat([x.unsqueeze(0), x_hyper.unsqueeze(0)], dim=2).squeeze(0)
        x_gnn = F.relu(self.conv1(x, edge_index))
        x_gnn = F.dropout(x_gnn, training=self.training)
        x_gnn = self.conv2(x_gnn, edge_index)
        x = self.lp(x_gnn)
        if self.is_regression:
            x = self.head(x).squeeze()
        # This is no longer needed, since F.log_softmax will be applied in train.py
        # else:
        #     x = F.log_softmax(x, dim=1)
        return x


class LPGCNEDGNNAblation(torch.nn.Module):
    def __init__(self, info, *args, **kwargs):
        super().__init__()
        dim = 64
        self.hyper = EquivSetGNN(info["num_node_features"], dim, *args)
        self.is_regression = info["is_regression"]
        # For LP-GCN+EDHNN, --Classifier_num_layers must be 2 and --Classifier_hidden for EDHNN must be 64
        if info["is_regression"]:
            self.conv1 = GCNConv(info["num_node_features"] + dim, dim)
            self.conv2 = GCNConv(dim, dim)
            self.lp = nn.Linear(dim, dim)
            self.head = nn.Linear(dim, 1)
        else:
            self.conv1 = GCNConv(info["num_node_features"] + dim, dim)
            self.conv2 = GCNConv(dim, info["num_classes"])
            self.lp = nn.Linear(info["num_classes"], info["num_classes"])

    def reset_parameters(self):
        self.hyper.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lp.reset_parameters()
        if self.is_regression:
            self.head.reset_parameters()

    def forward(self, data, *args, **kwargs):
        hgb_data, edgnn_data = data
        x, edge_index, hyperedge_index = hgb_data.x, hgb_data.edge_index, hgb_data.hyperedge_index
        x_hyper = self.hyper(edgnn_data)
        x = torch.cat([x.unsqueeze(0), x_hyper.unsqueeze(0)], dim=2).squeeze(0)
        x_gnn = F.relu(self.conv1(x, edge_index))
        x_gnn = F.dropout(x_gnn, training=self.training)
        x_gnn = self.conv2(x_gnn, edge_index)
        x = self.lp(x_gnn)
        if self.is_regression:
            x = self.head(x).squeeze()
        # This is no longer needed, since F.log_softmax will be applied in train.py
        # else:
        #     x = F.log_softmax(x, dim=1)
        return x
