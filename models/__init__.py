from models.allset import SetGNN
from models.average_ensemble import AverageEnsemble
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
}
