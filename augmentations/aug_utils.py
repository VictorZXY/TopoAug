import torch.nn as nn
import torch.nn.functional as F
import torch_geometric


class Compose(nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def forward(self, data):
        for aug in self.transforms:
            data = aug(data)
        return data


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        x = data.x
        y = data.y
        train_mask = data.train_mask
        test_mask = data.test_mask
        edge_idx = data.edge_index

        x = F.normalize(x)
        new_data = torch_geometric.data.Data(x=x, y=y, edge_index=edge_idx, train_mask=train_mask, test_mask=test_mask)
        return new_data
