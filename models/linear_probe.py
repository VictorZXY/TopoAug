import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, HypergraphConv, GAT

from models.edgnn import EquivSetGNN


class LPGCNHyperGCN(torch.nn.Module):
    def __init__(self, info, *args, **kwargs):
        super().__init__()
        dim = 64
        self.hyper1 = HypergraphConv(info["num_node_features"], dim)
        self.conv1 = GCNConv(info["num_node_features"], dim)
        self.is_regression = info["is_regression"]
        if info["is_regression"]:
            self.hyper2 = HypergraphConv(dim, dim)
            self.conv2 = GCNConv(dim, dim)
            self.lp = nn.Linear(2 * dim, dim)
            self.head = nn.Linear(dim, 1)
        else:
            self.conv2 = GCNConv(dim, info["num_classes"])
            self.hyper2 = HypergraphConv(dim, info["num_classes"])
            self.lp = nn.Linear(2 * info["num_classes"], info["num_classes"])

    def forward(self, data, *args, **kwargs):
        x, edge_index, hyperedge_index = data.x, data.edge_index, data.hyperedge_index
        x_gnn = F.relu(self.conv1(x, edge_index))
        x_hyper = F.relu(self.hyper1(x, hyperedge_index))
        x_gnn = F.dropout(x_gnn, training=self.training)
        x_hyper = F.dropout(x_hyper, training=self.training)
        x_gnn = self.conv2(x_gnn, edge_index)
        x_hyper = self.hyper2(x_hyper, hyperedge_index)
        x = self.lp(torch.cat([x_gnn.unsqueeze(0), x_hyper.unsqueeze(0)], dim=2)).squeeze(0)

        if self.is_regression:
            x = self.head(x).squeeze()
        # This is no longer needed, since F.log_softmax will be applied in train.py
        # else:
        #     x = F.log_softmax(x, dim=1)
        return x


class LPGATHyperGCN(torch.nn.Module):
    def __init__(self, info, *args, **kwargs):
        super().__init__()
        dim = 64
        self.hyper1 = HypergraphConv(info["num_node_features"], dim)
        self.conv1 = GAT(info["num_node_features"], dim, num_layers=1)
        self.is_regression = info["is_regression"]
        if info["is_regression"]:
            self.hyper2 = HypergraphConv(dim, dim)
            self.conv2 = GAT(dim, dim, num_layers=1)
            self.lp = nn.Linear(2 * dim, dim)
            self.head = nn.Linear(dim, 1)
        else:
            self.conv2 = GCNConv(dim, info["num_classes"])
            self.hyper2 = GAT(dim, info["num_classes"], num_layers=1)
            self.lp = nn.Linear(2 * info["num_classes"], info["num_classes"])

    def forward(self, data, *args, **kwargs):
        x, edge_index, hyperedge_index = data.x, data.edge_index, data.hyperedge_index
        x_gnn = F.relu(self.conv1(x, edge_index))
        x_hyper = F.relu(self.hyper1(x, hyperedge_index))
        x_gnn = F.dropout(x_gnn, training=self.training)
        x_hyper = F.dropout(x_hyper, training=self.training)
        x_gnn = self.conv2(x_gnn, edge_index)
        x_hyper = self.hyper2(x_hyper, hyperedge_index)
        x = self.lp(torch.cat([x_gnn.unsqueeze(0), x_hyper.unsqueeze(0)], dim=2)).squeeze(0)

        if self.is_regression:
            x = self.head(x).squeeze()
        # This is no longer needed, since F.log_softmax will be applied in train.py
        # else:
        #     x = F.log_softmax(x, dim=1)
        return x


class LPGCNEDGNN(torch.nn.Module):
    def __init__(self, info, *args, **kwargs):
        super().__init__()
        dim = 64
        self.is_regression = info["is_regression"]
        # For LP-GCN+EDHNN, --Classifier_num_layers must be 2 and --Classifier_hidden for EDHNN must be 64
        if info["is_regression"]:
            self.hyper = EquivSetGNN(info["num_node_features"], dim, *args)
            self.conv1 = GCNConv(info["num_node_features"], dim)
            self.conv2 = GCNConv(dim, dim)
            self.lp = nn.Linear(2 * dim, dim)
            self.head = nn.Linear(dim, 1)
        else:
            self.hyper = EquivSetGNN(info["num_node_features"], info["num_classes"], *args)
            self.conv1 = GCNConv(info["num_node_features"], dim)
            self.conv2 = GCNConv(dim, info["num_classes"])
            self.lp = nn.Linear(2 * info["num_classes"], info["num_classes"])

    def reset_parameters(self):
        self.hyper.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lp.reset_parameters()
        if self.is_regression:
            self.head.reset_parameters()

    def forward(self, data, *args, **kwargs):
        hgb_data, edgnn_data = data
        x_gnn, edge_index, hyperedge_index = hgb_data.x, hgb_data.edge_index, hgb_data.hyperedge_index
        x_gnn = F.relu(self.conv1(x_gnn, edge_index))
        x_gnn = F.dropout(x_gnn, training=self.training)
        x_gnn = self.conv2(x_gnn, edge_index)
        x_hyper = self.hyper(edgnn_data)
        x = self.lp(torch.cat([x_gnn.unsqueeze(0), x_hyper.unsqueeze(0)], dim=2)).squeeze(0)

        if self.is_regression:
            x = self.head(x).squeeze()
        # This is no longer needed, since F.log_softmax will be applied in train.py
        # else:
        #     x = F.log_softmax(x, dim=1)
        return x


class LPGATEDGNN(torch.nn.Module):
    def __init__(self, info, *args, **kwargs):
        super().__init__()
        dim = 64
        self.is_regression = info["is_regression"]
        # For LP-GAT+EDHNN, --Classifier_num_layers must be 2 and --Classifier_hidden for EDHNN must be 64
        if info["is_regression"]:
            self.hyper = EquivSetGNN(info["num_node_features"], dim, *args)
            self.conv1 = GAT(info["num_node_features"], dim, num_layers=1)
            self.conv2 = GAT(dim, dim, num_layers=1)
            self.lp = nn.Linear(2 * dim, dim)
            self.head = nn.Linear(dim, 1)
        else:
            self.hyper = EquivSetGNN(info["num_node_features"], info["num_classes"], *args)
            self.conv1 = GAT(info["num_node_features"], dim, num_layers=1)
            self.conv2 = GAT(dim, info["num_classes"], num_layers=1)
            self.lp = nn.Linear(2 * info["num_classes"], info["num_classes"])

    def reset_parameters(self):
        self.hyper.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lp.reset_parameters()
        if self.is_regression:
            self.head.reset_parameters()

    def forward(self, data, *args, **kwargs):
        hgb_data, edgnn_data = data
        x_gnn, edge_index, hyperedge_index = hgb_data.x, hgb_data.edge_index, hgb_data.hyperedge_index
        x_gnn = F.relu(self.conv1(x_gnn, edge_index))
        x_gnn = F.dropout(x_gnn, training=self.training)
        x_gnn = self.conv2(x_gnn, edge_index)
        x_hyper = self.hyper(edgnn_data)
        x = self.lp(torch.cat([x_gnn.unsqueeze(0), x_hyper.unsqueeze(0)], dim=2)).squeeze(0)

        if self.is_regression:
            x = self.head(x).squeeze()
        # This is no longer needed, since F.log_softmax will be applied in train.py
        # else:
        #     x = F.log_softmax(x, dim=1)
        return x


class LPGATGCN(torch.nn.Module):
    def __init__(self, info, *args, **kwargs):
        super().__init__()
        dim = 64
        self.hyper1 = GAT(info["num_node_features"], dim, num_layers=1)
        self.conv1 = GCNConv(info["num_node_features"], dim)
        self.is_regression = info["is_regression"]
        if info["is_regression"]:
            self.hyper2 = GAT(dim, dim, num_layers=1)
            self.conv2 = GCNConv(dim, dim)
            self.lp = nn.Linear(2 * dim, dim)
            self.head = nn.Linear(dim, 1)
        else:
            self.conv2 = GCNConv(dim, info["num_classes"])
            self.hyper2 = GAT(dim, info["num_classes"], num_layers=1)
            self.lp = nn.Linear(2 * info["num_classes"], info["num_classes"])

    def forward(self, data, *args, **kwargs):
        x, edge_index, hyperedge_index = data.x, data.edge_index, data.hyperedge_index
        x_gnn = F.relu(self.conv1(x, edge_index))
        x_hyper = F.relu(self.hyper1(x, edge_index))
        x_gnn = F.dropout(x_gnn, training=self.training)
        x_hyper = F.dropout(x_hyper, training=self.training)
        x_gnn = self.conv2(x_gnn, edge_index)
        x_hyper = self.hyper2(x_hyper, edge_index)
        x = self.lp(torch.cat([x_gnn.unsqueeze(0), x_hyper.unsqueeze(0)], dim=2)).squeeze(0)

        if self.is_regression:
            x = self.head(x).squeeze()
        # This is no longer needed, since F.log_softmax will be applied in train.py
        # else:
        #     x = F.log_softmax(x, dim=1)
        return x


class LPGATGAT(torch.nn.Module):
    def __init__(self, info, *args, **kwargs):
        super().__init__()
        dim = 64
        self.hyper1 = GAT(info["num_node_features"], dim, num_layers=1)
        self.conv1 = GAT(info["num_node_features"], dim, num_layers=1)
        self.is_regression = info["is_regression"]
        if info["is_regression"]:
            self.hyper2 = GAT(dim, dim, num_layers=1)
            self.conv2 = GAT(dim, dim, num_layers=1)
            self.lp = nn.Linear(2 * dim, dim)
            self.head = nn.Linear(dim, 1)
        else:
            self.conv2 = GAT(dim, info["num_classes"], num_layers=1)
            self.hyper2 = GAT(dim, info["num_classes"], num_layers=1)
            self.lp = nn.Linear(2 * info["num_classes"], info["num_classes"])

    def forward(self, data, *args, **kwargs):
        x, edge_index, hyperedge_index = data.x, data.edge_index, data.hyperedge_index
        x_gnn = F.relu(self.conv1(x, edge_index))
        x_hyper = F.relu(self.hyper1(x, edge_index))
        x_gnn = F.dropout(x_gnn, training=self.training)
        x_hyper = F.dropout(x_hyper, training=self.training)
        x_gnn = self.conv2(x_gnn, edge_index)
        x_hyper = self.hyper2(x_hyper, edge_index)
        x = self.lp(torch.cat([x_gnn.unsqueeze(0), x_hyper.unsqueeze(0)], dim=2)).squeeze(0)

        if self.is_regression:
            x = self.head(x).squeeze()
        # This is no longer needed, since F.log_softmax will be applied in train.py
        # else:
        #     x = F.log_softmax(x, dim=1)
        return x


class LPGCNGCN(torch.nn.Module):
    def __init__(
            self, info, *args, **kwargs):
        super().__init__()
        dim = 64
        self.hyper1 = GCNConv(info["num_node_features"], dim)
        self.conv1 = GCNConv(info["num_node_features"], dim)
        self.is_regression = info["is_regression"]
        if info["is_regression"]:
            self.hyper2 = GCNConv(dim, dim)
            self.conv2 = GCNConv(dim, dim)
            self.lp = nn.Linear(2 * dim, dim)
            self.head = nn.Linear(dim, 1)
        else:
            self.hyper2 = GCNConv(dim, info["num_classes"])
            self.conv2 = GCNConv(dim, info["num_classes"])
            self.lp = nn.Linear(2 * info["num_classes"], info["num_classes"])

    def forward(self, data, *args, **kwargs):
        x, edge_index, hyperedge_index = data.x, data.edge_index, data.hyperedge_index
        x_gnn = F.relu(self.conv1(x, edge_index))
        x_hyper = F.relu(self.hyper1(x, edge_index))
        x_gnn = F.dropout(x_gnn, training=self.training)
        x_hyper = F.dropout(x_hyper, training=self.training)
        x_gnn = self.conv2(x_gnn, edge_index)
        x_hyper = self.hyper2(x_hyper, edge_index)
        x = self.lp(torch.cat([x_gnn.unsqueeze(0), x_hyper.unsqueeze(0)], dim=2)).squeeze(0)

        if self.is_regression:
            x = self.head(x).squeeze()
        # This is no longer needed, since F.log_softmax will be applied in train.py
        # else:
        #     x = F.log_softmax(x, dim=1)
        return x


class LPHyperHyper(torch.nn.Module):
    def __init__(
            self, info, *args, **kwargs):
        super().__init__()
        dim = 64
        self.hyper1 = HypergraphConv(info["num_node_features"], dim)
        self.conv1 = HypergraphConv(info["num_node_features"], dim)
        self.is_regression = info["is_regression"]
        if info["is_regression"]:
            self.hyper2 = HypergraphConv(dim, dim)
            self.conv2 = HypergraphConv(dim, dim)
            self.lp = nn.Linear(2 * dim, dim)
            self.head = nn.Linear(dim, 1)
        else:
            self.conv2 = HypergraphConv(dim, info["num_classes"])
            self.hyper2 = HypergraphConv(dim, info["num_classes"])
            self.lp = nn.Linear(2 * info["num_classes"], info["num_classes"])

    def forward(self, data, *args, **kwargs):
        x, edge_index, hyperedge_index = data.x, data.edge_index, data.hyperedge_index
        x_gnn = F.relu(self.conv1(x, hyperedge_index))
        x_hyper = F.relu(self.hyper1(x, hyperedge_index))
        x_gnn = F.dropout(x_gnn, training=self.training)
        x_hyper = F.dropout(x_hyper, training=self.training)
        x_gnn = self.conv2(x_gnn, hyperedge_index)
        x_hyper = self.hyper2(x_hyper, hyperedge_index)
        x = self.lp(torch.cat([x_gnn.unsqueeze(0), x_hyper.unsqueeze(0)], dim=2)).squeeze(0)

        if self.is_regression:
            x = self.head(x).squeeze()
        # This is no longer needed, since F.log_softmax will be applied in train.py
        # else:
        #     x = F.log_softmax(x, dim=1)
        return x


class LPEDGNNHyper(torch.nn.Module):
    def __init__(self, info, *args, **kwargs):
        super().__init__()
        dim = 64
        self.is_regression = info["is_regression"]
        # For LP-EDHNN+HyperConv, --Classifier_num_layers must be 2 and --Classifier_hidden for EDHNN must be 64
        if info["is_regression"]:
            self.hyper = EquivSetGNN(info["num_node_features"], dim, *args)
            self.conv1 = HypergraphConv(info["num_node_features"], dim)
            self.conv2 = HypergraphConv(dim, dim)
            self.lp = nn.Linear(2 * dim, dim)
            self.head = nn.Linear(dim, 1)
        else:
            self.hyper = EquivSetGNN(info["num_node_features"], info["num_classes"], *args)
            self.conv1 = HypergraphConv(info["num_node_features"], dim)
            self.conv2 = HypergraphConv(dim, info["num_classes"])
            self.lp = nn.Linear(2 * info["num_classes"], info["num_classes"])

    def reset_parameters(self):
        self.hyper.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lp.reset_parameters()
        if self.is_regression:
            self.head.reset_parameters()

    def forward(self, data, *args, **kwargs):
        hgb_data, edgnn_data = data
        x_gnn, edge_index, hyperedge_index = hgb_data.x, hgb_data.edge_index, hgb_data.hyperedge_index
        x_gnn = F.relu(self.conv1(x_gnn, edge_index))
        x_gnn = F.dropout(x_gnn, training=self.training)
        x_gnn = self.conv2(x_gnn, edge_index)
        x_hyper = self.hyper(edgnn_data)
        x = self.lp(torch.cat([x_gnn.unsqueeze(0), x_hyper.unsqueeze(0)], dim=2)).squeeze(0)

        if self.is_regression:
            x = self.head(x).squeeze()
        # This is no longer needed, since F.log_softmax will be applied in train.py
        # else:
        #     x = F.log_softmax(x, dim=1)
        return x


class LPEDGNNEDGNN(torch.nn.Module):
    def __init__(self, info, *args, **kwargs):
        super().__init__()
        dim = 64
        self.is_regression = info["is_regression"]
        # For LP-EDHNN+EDHNN, --Classifier_num_layers must be 2 and --Classifier_hidden for EDHNN must be 64
        if info["is_regression"]:
            self.hyper = EquivSetGNN(info["num_node_features"], dim, *args)
            self.conv = EquivSetGNN(info["num_node_features"], dim, *args)
            self.lp = nn.Linear(2 * dim, dim)
            self.head = nn.Linear(dim, 1)
        else:
            self.hyper = EquivSetGNN(info["num_node_features"], info["num_classes"], *args)
            self.conv = EquivSetGNN(info["num_node_features"], info["num_classes"], *args)
            self.lp = nn.Linear(2 * info["num_classes"], info["num_classes"])

    def reset_parameters(self):
        self.hyper.reset_parameters()
        self.conv.reset_parameters()
        self.lp.reset_parameters()
        if self.is_regression:
            self.head.reset_parameters()

    def forward(self, data, *args, **kwargs):
        x_gnn = self.conv(data)
        x_hyper = self.hyper(data)
        x = self.lp(torch.cat([x_gnn.unsqueeze(0), x_hyper.unsqueeze(0)], dim=2)).squeeze(0)

        if self.is_regression:
            x = self.head(x).squeeze()
        # This is no longer needed, since F.log_softmax will be applied in train.py
        # else:
        #     x = F.log_softmax(x, dim=1)
        return x
