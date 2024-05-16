import torch
from deepspeed.utils.config import *
from deepspeed import comm as dist
from deepspeed.runtime import DeepSpeedEngine
from deepspeed.comm import init_distributed


def add_core_arguments(parser):
    r"""Helper (internal) function to update an argument parser with an argument group of the core DeepSpeed arguments.
        The core set of DeepSpeed arguments include the following:
        1) --deepspeed: boolean flag to enable DeepSpeed
        2) --deepspeed_config <json file path>: path of a json configuration file to configure DeepSpeed runtime.

        This is a helper function to the public add_config_arguments()

    Arguments:
        parser: argument parser
    Return:
        parser: Updated Parser
    """
    group = parser.add_argument_group('DeepSpeed', 'DeepSpeed configurations')

    group.add_argument('--deepspeed',
                       default=False,
                       action='store_true',
                       help='Enable DeepSpeed (helper flag for user code, no impact on DeepSpeed backend)')

    group.add_argument('--deepspeed_config', default=None, type=str, help='DeepSpeed json configuration file.')

    group.add_argument('--deepscale',
                       default=False,
                       action='store_true',
                       help='Deprecated enable DeepSpeed (helper flag for user code, no impact on DeepSpeed backend)')

    group.add_argument('--deepscale_config',
                       default=None,
                       type=str,
                       help='Deprecated DeepSpeed json configuration file.')

    group.add_argument('--deepspeed_mpi',
                       default=False,
                       action='store_true',
                       help="Run via MPI, this will attempt to discover the necessary variables to initialize torch "
                            "distributed from the MPI environment")

    return parser


def initialize(args=None,
               model=None,
               optimizer=None,
               model_parameters=None,
               training_data=None):

    assert model is not None, "deepspeed.initialize requires a model"

    dist_backend = Communication_backend_name
    dist.init_distributed(dist_backend=dist_backend)
    engine = DeepSpeedEngine(model=model,
                             optimizer=optimizer,
                             model_parameters=model_parameters,
                             training_data=training_data)

    return_items = [engine, engine.optimizer, engine.training_dataloader, engine.lr_scheduler]
    return tuple(return_items)
