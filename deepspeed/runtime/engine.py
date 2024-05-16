import torch
from .dataloader import DeepSpeedDataLoader

from deepspeed import comm as dist
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.zero.optimizer_stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.runtime import lr_schedules
from deepspeed.utils.constants import *
from deepspeed.utils.config import *

GROUP = None
class DeepSpeedEngine(torch.nn.Module):
    def __init__(
            self,
            model,
            optimizer=None,
            model_parameters=None,
            training_data=None):
        super(DeepSpeedEngine, self).__init__()
        self.local_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        global GROUP
        GROUP = dist.new_group(list(range(self.world_size)))
        # self.global_rank = dist.get_global_rank()

        self.device = torch.device('cuda:{}'.format(self.local_rank))
        # dataloader
        self.training_dataloader = self._configure_distributed_data(training_data)

        # model
        self._configure_distributed_model(model)

        # optimizer
        self.param_names = {param: name for name, param in model.named_parameters()}
        self._configure_optimizer(optimizer, model_parameters)

        # lr_scheduler
        self._configure_lr_scheduler()

    # # # dataloader # # #
    def _configure_distributed_data(self, dataset):
        return DeepSpeedDataLoader(dataset=dataset,
                                   batch_size=Train_micro_batch_size_per_gpu,
                                   local_rank=self.local_rank)

    # # # model # # #
    def _configure_distributed_model(self, model):
        modules = self.__dict__.get('_modules')
        modules['module'] = model
        self.__dict__['module'] = model
        self.module.half()
        self.module.to(self.device)
        # for p in self.module.parameters():
        #     if torch.is_tensor(p):
        #         dist.broadcast(p, self.local_rank, group=GROUP)

    # # # optimizer # # #
    def _configure_zero_optimizer(self, optimizer):
        optimizer = DeepSpeedZeroOptimizer(
            optimizer,
            self.param_names,
            dynamic_loss_scale=True,
            dynamic_loss_args=None,
            clip_grad=0,
            dp_process_group=dist.new_group(ranks=range(dist.get_world_size())),
            mpu=None,
            postscale_gradients=True,
            gradient_predivide_factor=1)
        return optimizer

    def _configure_optimizer(self, client_optimizer, model_parameters):
        if client_optimizer is None:
            raise ValueError('not support "None" optimizer')
            # basic_optimizer = DeepSpeedCPUAdam(model_parameters, adamw_mode=True)
        else:
            assert isinstance(client_optimizer, torch.optim.Optimizer), "optimizer is not torch.optim.Optimizer"
            basic_optimizer = client_optimizer

        basic_optimizer.param_groups[:] = [pg for pg in basic_optimizer.param_groups if len(pg["params"]) != 0]

        self.basic_optimizer = basic_optimizer

        self.optimizer = self._configure_zero_optimizer(basic_optimizer)

    # # # lr_scheduler # # #
    def _scheduler_from_config(self, optimizer):
        scheduler_name = Scheduler_name
        if scheduler_name is not None:
            if hasattr(lr_schedules, scheduler_name):
                scheduler = getattr(lr_schedules, scheduler_name)
            else:
                scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)

            scheduler_params = Scheduler_config
            instantiated_scheduler = scheduler(optimizer, **scheduler_params)
            return instantiated_scheduler
        else:
            return None

    def _configure_lr_scheduler(self):
        lr_scheduler = self._scheduler_from_config(self.optimizer)
        self.lr_scheduler = lr_scheduler

    # # # inner interface # # #
    def _take_model_step(self):
        self.optimizer.step()
        self._global_grad_norm = self.optimizer._global_grad_norm
        overflow = self.optimizer.overflow
        self._step_applied = not overflow

        self.losses = 0.0
        self.global_steps += 1
        self.global_samples += self.train_batch_size()

    def _scale_loss_by_gas(self, prescaled_loss, eval_micro_batches=None):
        # 1
        scaling_factor = self.gradient_accumulation_steps() if eval_micro_batches is None else eval_micro_batches
        scaled_loss = prescaled_loss / scaling_factor
        return scaled_loss

    def allreduce_gradients(self):
        self.optimizer.is_gradient_accumulation_boundary = self.is_gradient_accumulation_boundary()
        self.optimizer.overlapping_partition_gradients_reduce_epilogue()

    # # # public interface # # #
    def forward(self, *inputs, **kwargs):
        loss = self.module(*inputs, **kwargs)
        return loss

    def step(self):
        self._step_applied = False
        self.gas_boundary_ctr += 1
        self._take_model_step()
        self.micro_steps += 1

    def backward(self, loss, allreduce_gradients=True, release_loss=False, retain_graph=False, scale_wrt_gas=True):
        loss = self._scale_loss_by_gas(loss.float())
        self.losses += loss.mean().item()
        self.optimizer.is_gradient_accumulation_boundary = self.is_gradient_accumulation_boundary()
        self.optimizer.backward(loss, retain_graph=retain_graph)
        self.allreduce_gradients()

        return loss

    def train(self, mode=True):
        self.module.train(mode)

    def eval(self):
        self.module.train(False)
