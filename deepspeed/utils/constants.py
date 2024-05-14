from enum import Enum

NCCL_BACKEND = 'nccl'
CCL_BACKEND = 'ccl'
MPI_BACKEND = 'mpi'
GLOO_BACKEND = 'gloo'
HCCL_BACKEND = 'hccl'

ENABLED_DEFAULT = False

#############################################
# Routes
#############################################
ROUTE_TRAIN = "train"
ROUTE_EVAL = "eval"
ROUTE_PREDICT = "predict"
ROUTE_ENCODE = "encode"

#############################################
# Optimizer
#############################################
ZERO_OPTIMIZATION = "zero_optimization"


class ReduceOp(Enum):
    SUM = 0
    PRODUCT = 1
    MIN = 2
    MAX = 3
    BAND = 4
    BOR = 5
    BXOR = 6
    AVG = 7
    UNUSED = 8


class ZeroParamStatus(Enum):
    AVAILABLE = 1
    NOT_AVAILABLE = 2
    INFLIGHT = 3


class ZeroStageEnum(int, Enum):
    disabled = 0
    optimizer_states = 1
    gradients = 2
    weights = 3
    max_stage = 3
