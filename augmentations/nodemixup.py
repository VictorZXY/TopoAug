import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric


class NodeMixUp(nn.Module):
    def __init__(self, lamb, classes):
        super().__init__()
        self.lamb = lamb
        self.classes = classes

    def forward(self, data):
        x = data.x
        y = data.y
        edge_idx = data.edge_index

        n, d = x.shape

        pair_idx = torch.randperm(n)
        x_b = x[pair_idx]
        y_b = y[pair_idx]
        y_a_oh = F.one_hot(y, self.classes)
        y_b_oh = F.one_hot(y_b, self.classes)

        x_mix = (self.lamb * x) + (1 - self.lamb) * x_b
        y_mix = (self.lamb * y_a_oh) + (1 - self.lamb) * y_b_oh
        new_y = y_mix.argmax(1)

        # new_x = torch.vstack([x, x_mix])
        # new_y = torch.vstack([y_a_oh, y_mix])

        new_data = torch_geometric.data.Data(x=x_mix, y=new_y, edge_index=edge_idx)
        return new_data
