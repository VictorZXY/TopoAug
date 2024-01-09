import torch

from models.gat import GATNet, GATv2Net
from models.gcn import GCNNet
from models.graphsage import SAGENet
from models.hyper_conv_atten import HyperConv, HyperAtten

mapstr2model = {
    'gcn': GCNNet,
    'sage': SAGENet,
    'gat': GATNet,
    'gatv2': GATv2Net,
    'hyper-gcn': HyperConv,
    'hyper-gat': HyperAtten,
}


def remove_model_prefix(d):
    new_dict = {}
    for key, value in d.items():
        new_key = key.replace(
            "model.", "", 1
        )  # Replace the first occurrence of "model."
        new_dict[new_key] = value
    return new_dict


def plt_model_load(model, checkpoint):
    state_dict = remove_model_prefix(torch.load(checkpoint)['state_dict'])
    model.load_state_dict(state_dict)
    return model


class AverageEnsemble(torch.nn.Module):
    def __init__(self, info, *args, **kwargs):
        super().__init__()
        self.model1 = mapstr2model[info["model1"]](info)
        self.model2 = mapstr2model[info["model2"]](info)
        # initialize the two models with the checkpoint weights
        self.model1 = plt_model_load(self.model1, info["checkpoint1"])
        self.model2 = plt_model_load(self.model2, info["checkpoint2"])

    def forward(self, data, *args, **kwargs):
        x1 = self.model1(data, *args, **kwargs)
        x2 = self.model2(data, *args, **kwargs)
        return (x1 + x2) / 2
