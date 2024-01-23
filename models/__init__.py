from models.allset import SetGNN
from models.average_ensemble import AverageEnsemble
from models.cross_atten import CAGCNHyperConv, CASAGEHyperConv
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
    LPGCNHyperConv, LPGATHyperConv, LPSAGEHyperConv, LPGCNEDGNN, LPGATEDGNN, LPSAGEEDGNN,
    LPGCNGCN, LPGATGAT, LPGATGCN, LPSAGEGAT, LPSAGEGCN, LPSAGESAGE,
    LPHyperConvHyperConv, LPEDGNNHyperConv, LPEDGNNEDGNN
)
from models.unigcn2 import UniGCNII

factory = {
    'gcn': GCNNet,
    'gat': GATNet,
    'gatv2': GATv2Net,
    'sage': SAGENet,
    'hyperconv': HyperConv,
    'hyperatten': HyperAtten,
    'ensemble': AverageEnsemble,
    'ogb-gcn': OGBGCN,
    'ogb-sage': OGBSAGE,
    'ogb-hypersage': OGBHyperSAGE,
    'diff': DiffPool,
    'edgnn': EquivSetGNN,
    'lp-gcn-hyperconv': LPGCNHyperConv,
    'lp-gat-hyperconv': LPGATHyperConv,
    'lp-sage-hyperconv': LPSAGEHyperConv,
    'lp-gcn-edhnn': LPGCNEDGNN,
    'lp-gat-edhnn': LPGATEDGNN,
    'lp-sage-edhnn': LPSAGEEDGNN,
    'lp-sage-gcn': LPSAGEGCN,
    'lp-sage-gat': LPSAGEGAT,
    'lp-sage-sage': LPSAGESAGE,
    'lp-gat-gcn': LPGATGCN,
    'lp-gat-gat': LPGATGAT,
    'lp-gcn-gcn': LPGCNGCN,
    'lp-hyperconv-hyperconv': LPHyperConvHyperConv,
    'lp-edhnn-hyperconv': LPEDGNNHyperConv,
    'lp-edhnn-edhnn': LPEDGNNEDGNN,
    'ca-gcn-hyperconv': CAGCNHyperConv,
    'ca-sage-hyperconv': CASAGEHyperConv
}
