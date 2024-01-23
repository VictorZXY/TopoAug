import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAT, SAGEConv, HypergraphConv

from models.attention import Attention


class CAGCNHyperConv(torch.nn.Module):
    def __init__(self, info, *args, **kwargs):
        super().__init__()
        dim = 32
        self.hyper1 = HypergraphConv(info["num_node_features"], dim)
        self.conv1 = GCNConv(info["num_node_features"], dim)
        self.attn1 = Attention(dim)
        self.is_regression = info["is_regression"]
        self.is_edge_pred = info["is_edge_pred"]
        if info["is_regression"]:
            self.hyper2 = HypergraphConv(dim, dim)
            self.conv2 = GCNConv(dim, dim)
            self.attn2 = Attention(dim)
            self.head = nn.Linear(dim, 1)
        elif info["is_edge_pred"]:
            self.hyper2 = HypergraphConv(dim, dim)
            self.conv2 = GCNConv(dim, dim)
            self.attn2 = Attention(dim)
        else:
            self.conv2 = GCNConv(dim, info["num_classes"])
            self.hyper2 = HypergraphConv(dim, info["num_classes"])
            self.attn2 = Attention(info["num_classes"], head=1)

    def reset_parameters(self):
        self.hyper1.reset_parameters()
        self.hyper2.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.attn1.reset_parameters()
        self.attn2.reset_parameters()
        if self.is_regression:
            self.head.reset_parameters()

    def forward(self, data, *args, **kwargs):
        x, edge_index, hyperedge_index = data.x, data.edge_index, data.hyperedge_index
        if self.is_edge_pred:
            edge_index = args[0][0]  # the message passing edge index
        x_gnn = F.relu(self.conv1(x, edge_index))
        x_hyper = F.relu(self.hyper1(x, hyperedge_index))
        x = self.attn1(x_gnn.unsqueeze(0), x_hyper.unsqueeze(0)).squeeze(0)
        x = F.dropout(x, training=self.training)
        x_gnn = self.conv2(x, edge_index)
        x_hyper = self.hyper2(x_hyper, hyperedge_index)
        x = self.attn2(x_gnn.unsqueeze(0), x_hyper.unsqueeze(0)).squeeze(0)

        if self.is_regression:
            x = self.head(x).squeeze()
        elif self.is_edge_pred:
            # args[1] is the edge_label_index
            edge_label_index = args[0][1]
            x = x[edge_label_index[0]] * x[edge_label_index[1]]
            x = torch.sum(x, -1)
        # This is no longer needed, since F.log_softmax will be applied in train.py
        # else:
        #     x = F.log_softmax(x, dim=1)
        return x


class CAGATHyperConv(torch.nn.Module):
    def __init__(self, info, *args, **kwargs):
        super().__init__()
        dim = 32
        self.hyper1 = HypergraphConv(info["num_node_features"], dim)
        self.conv1 = GAT(info["num_node_features"], dim, num_layers=1)
        self.attn1 = Attention(dim)
        self.is_regression = info["is_regression"]
        self.is_edge_pred = info["is_edge_pred"]
        if info["is_regression"]:
            self.hyper2 = HypergraphConv(dim, dim)
            self.conv2 = GAT(dim, dim, num_layers=1)
            self.attn2 = Attention(dim)
            self.head = nn.Linear(dim, 1)
        elif info["is_edge_pred"]:
            self.hyper2 = HypergraphConv(dim, dim)
            self.conv2 = GAT(dim, dim, num_layers=1)
            self.attn2 = Attention(dim)
        else:
            self.conv2 = GAT(dim, info["num_classes"], num_layers=1)
            self.hyper2 = HypergraphConv(dim, info["num_classes"])
            self.attn2 = Attention(info["num_classes"], head=1)

    def reset_parameters(self):
        self.hyper1.reset_parameters()
        self.hyper2.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.attn1.reset_parameters()
        self.attn2.reset_parameters()
        if self.is_regression:
            self.head.reset_parameters()

    def forward(self, data, *args, **kwargs):
        x, edge_index, hyperedge_index = data.x, data.edge_index, data.hyperedge_index
        if self.is_edge_pred:
            edge_index = args[0][0]  # the message passing edge index
        x_gnn = F.relu(self.conv1(x, edge_index))
        x_hyper = F.relu(self.hyper1(x, hyperedge_index))
        x = self.attn1(x_gnn.unsqueeze(0), x_hyper.unsqueeze(0)).squeeze(0)
        x = F.dropout(x, training=self.training)
        x_gnn = self.conv2(x, edge_index)
        x_hyper = self.hyper2(x_hyper, hyperedge_index)
        x = self.attn2(x_gnn.unsqueeze(0), x_hyper.unsqueeze(0)).squeeze(0)

        if self.is_regression:
            x = self.head(x).squeeze()
        elif self.is_edge_pred:
            # args[1] is the edge_label_index
            edge_label_index = args[0][1]
            x = x[edge_label_index[0]] * x[edge_label_index[1]]
            x = torch.sum(x, -1)
        # This is no longer needed, since F.log_softmax will be applied in train.py
        # else:
        #     x = F.log_softmax(x, dim=1)
        return x


class CASAGEHyperConv(torch.nn.Module):
    def __init__(
            self, info, *args, **kwargs):
        super().__init__()
        dim = 32
        self.hyper1 = HypergraphConv(info["num_node_features"], dim)
        self.conv1 = SAGEConv(info["num_node_features"], dim, normalize=False)
        self.attn1 = Attention(dim)
        self.is_regression = info["is_regression"]
        self.is_edge_pred = info["is_edge_pred"]
        if info["is_regression"]:
            self.hyper2 = HypergraphConv(dim, dim)
            self.conv2 = SAGEConv(dim, dim, normalize=False)
            self.attn2 = Attention(dim)
            self.head = nn.Linear(dim, 1)
        elif info["is_edge_pred"]:
            self.hyper2 = HypergraphConv(dim, dim)
            self.conv2 = SAGEConv(dim, dim, normalize=False)
            self.attn2 = Attention(dim)
        else:
            self.conv2 = SAGEConv(dim, info["num_classes"], normalize=False)
            self.hyper2 = HypergraphConv(dim, info["num_classes"])
            self.attn2 = Attention(info["num_classes"], head=1)

    def reset_parameters(self):
        self.hyper1.reset_parameters()
        self.hyper2.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.attn1.reset_parameters()
        self.attn2.reset_parameters()
        if self.is_regression:
            self.head.reset_parameters()

    def forward(self, data, *args, **kwargs):
        x, edge_index, hyperedge_index = data.x, data.edge_index, data.hyperedge_index
        if self.is_edge_pred:
            edge_index = args[0][0]  # the message passing edge index
        x_gnn = F.relu(self.conv1(x, edge_index))
        x_hyper = F.relu(self.hyper1(x, hyperedge_index))
        x = self.attn1(x_gnn.unsqueeze(0), x_hyper.unsqueeze(0)).squeeze(0)
        x = F.dropout(x, training=self.training)
        x_gnn = self.conv2(x, edge_index)
        x_hyper = self.hyper2(x_hyper, hyperedge_index)
        x = self.attn2(x_gnn.unsqueeze(0), x_hyper.unsqueeze(0)).squeeze(0)

        if self.is_regression:
            x = self.head(x).squeeze()
        elif self.is_edge_pred:
            # args[1] is the edge_label_index
            edge_label_index = args[0][1]
            x = x[edge_label_index[0]] * x[edge_label_index[1]]
            x = torch.sum(x, -1)
        # This is no longer needed, since F.log_softmax will be applied in train.py
        # else:
        #     x = F.log_softmax(x, dim=1)
        return x
