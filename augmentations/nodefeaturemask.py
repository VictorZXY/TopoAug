import torch
import torch.nn as nn
import torch_geometric


class NodeFeatureMasking(nn.Module):
    def __init__(self, p=0.15):
        super().__init__()
        self.p = p

    def forward(self, data):
        x = data.x
        y = data.y
        edge_idx = data.edge_index

        n, d = x.shape

        idx = torch.empty((d,), dtype=torch.float32).uniform_(0, 1) < self.p
        x = x.clone()
        x[:, idx] = 0

        new_data = torch_geometric.data.Data(x=x, y=y, edge_index=edge_idx)
        return new_data
