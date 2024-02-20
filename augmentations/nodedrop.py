import torch
import torch.nn as nn
import torch_geometric


class NodeDrop(nn.Module):
    def __init__(self, p=0.05):
        super().__init__()
        self.p = p

    def forward(self, data):
        x = data.x
        y = data.y
        edge_idx = data.edge_index

        # Generate a mask for nodes to keep
        keep_mask = torch.empty(x.size(0)).uniform_(0, 1) >= self.p

        # Update node features and labels
        x = x[keep_mask]
        y = y[keep_mask]

        # Create a mapping for old to new node indices
        node_idx_mapping = torch.full((data.x.size(0),), -1, dtype=torch.long)
        node_idx_mapping[keep_mask] = torch.arange(x.size(0))

        # Update edge_index
        edge_idx = node_idx_mapping[edge_idx]
        # Remove edges that now have invalid indices
        edge_idx = edge_idx[:, (edge_idx >= 0).all(dim=0)]

        new_data = torch_geometric.data.Data(x=x, edge_index=edge_idx, y=y)

        return new_data
