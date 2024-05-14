import torch
import os

from .torch import TorchBackend

from deepspeed.utils.config import *
from deepspeed.utils.constants import *
from deepspeed.utils import logger
from datetime import timedelta

cdb = None


def is_initialized():
    # assert cdb is not None, 'DeepSpeed backend not set, please initialize it using init_process_group()'
    if cdb is None:
        return False
    else:
        return cdb.is_initialized()


def all_reduce(tensor,
               op=ReduceOp.SUM,
               group=None,
               async_op=False,
               prof=False,
               log_name='all_reduce',
               debug=None):
    global cdb
    return cdb.all_reduce(tensor, op, group, async_op)


def new_group(ranks):
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    return cdb.new_group(ranks)


def broadcast(tensor, src, group=None, async_op=False):
    global cdb
    return cdb.broadcast(tensor=tensor, src=src, group=group, async_op=async_op)


def get_world_size(group=None):
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    return cdb.get_world_size(group)


def get_rank(group=None):
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    return cdb.get_rank(group)


def get_global_rank(group=None, group_rank=0):
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    return cdb.get_global_rank(group, group_rank)


def init_distributed(dist_backend=Communication_backend_name,
                     auto_mpi_discovery=True,
                     distributed_port=Distributed_port,
                     verbose=True,
                     timeout=Default_pg_timeout,
                     init_method=None,
                     rank=-1,
                     world_size=-1):
    global cdb
    assert isinstance(timeout, timedelta)
    required_env = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    if auto_mpi_discovery and not all(map(lambda v: v in os.environ, required_env)):
        mpi_discovery(distributed_port=distributed_port, verbose=verbose)
    cdb = TorchBackend(dist_backend, timeout, init_method, rank, world_size)
    logger.info('Initializing TorchBackend in DeepSpeed with backend {}'.format(dist_backend))


def mpi_discovery(distributed_port=Distributed_port, verbose=True):
    logger.info('do not support mpi yet')
    pass
