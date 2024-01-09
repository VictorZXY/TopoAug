from models.allset import SetGNN
from models.average_ensemble import AverageEnsemble
from models.cross_atten import CAGCNHyperGCN, CASAGEHyperGCN
from models.diffpool import DiffPool
from models.edgnn import EquivSetGNN
from models.gat import GATNet, GATv2Net
from models.gcn import GCNNet, OGBGCN
from models.graphsage import SAGENet, OGBSAGE
from models.hcha import HCHA
from models.hnhn import HNHN
from models.hyper_conv_atten import HyperConv, HyperAtten
from models.hypergcn import HyperGCN
from models.hypernd import HyperND
from models.hypersage import HyperSAGE, OGBHyperSAGE
from models.legcn import LEGCN
from models.linear_probe import (
    LPGCNHyperGCN, LPGATHyperGCN, LPGCNEDGNN, LPGATEDGNN, LPGCNGCN, LPGATGAT, LPGATGCN,
    LPHyperHyper, LPEDGNNHyper, LPEDGNNEDGNN
)
from models.unigcn2 import UniGCNII

factory = {
    'gcn': GCNNet,
    'sage': SAGENet,
    'gat': GATNet,
    'gatv2': GATv2Net,
    'hyper-gcn': HyperGCN,
    'hyper-gat': HyperAtten,
    'ensemble': AverageEnsemble,
    'ogb-gcn': OGBGCN,
    'ogb-sage': OGBSAGE,
    'ogb-hypersage': OGBHyperSAGE,
    'diff': DiffPool,
    'edgnn': EquivSetGNN,
    'lp-gcn-hyper-gcn': LPGCNHyperGCN,
    'lp-gat-hyper-gcn': LPGATHyperGCN,
    'lp-gcn-edhnn': LPGCNEDGNN,
    'lp-gat-edhnn': LPGATEDGNN,
    'lp-gat-gcn': LPGATGCN,
    'lp-gcn-gcn': LPGCNGCN,
    'lp-gat-gat': LPGATGAT,
    'lp-hyper-hyper': LPHyperHyper,
    'lp-edhnn-hyper': LPEDGNNHyper,
    'lp-edhnn-edhnn': LPEDGNNEDGNN,
    'ca-gcn-hyper-gcn': CAGCNHyperGCN,
    'ca-sage-hyper-gcn': CASAGEHyperGCN,
}
