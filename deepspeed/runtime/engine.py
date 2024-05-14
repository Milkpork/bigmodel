import torch
from .dataloader import DeepSpeedDataLoader

from deepspeed import comm as dist
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.zero.optimizer_stage_1_and_2 import DeepSpeedZeroOptimizer

from deepspeed.utils.constants import *
from deepspeed.utils.config import *


class DeepSpeedEngine(torch.nn.Module):
    def __init__(
            self,
            model,
            optimizer=None,
            model_parameters=None,
            training_data=None):
        super(DeepSpeedEngine, self).__init__()

        self.local_rank = dist.get_rank()
        self.global_rank = dist.get_world_size()
        self.world_size = dist.get_world_size()

        # dataloader
        self.training_dataloader = self._configure_distributed_data(training_data)

        # model
        self._configure_distributed_model(model)

        # optimizer
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
        self.module.to(self.device)

        for p in self.module.parameters():
            if torch.is_tensor(p):
                dist.broadcast(p, self.global_rank, group=dist.new_group(list(range(self.world_size))))

    # # # optimizer # # #
    def _configure_zero_optimizer(self, optimizer):
        optimizer = DeepSpeedZeroOptimizer(
            optimizer,
            self.param_names,
            dynamic_loss_scale=self.dynamic_loss_scale(),
            dynamic_loss_args=self.dynamic_loss_scale_args(),
            clip_grad=self.gradient_clipping(),
            dp_process_group=self.seq_data_parallel_group,
            mpu=self.mpu,
            postscale_gradients=self.postscale_gradients(),
            gradient_predivide_factor=self.gradient_predivide_factor())
        return optimizer

    def _configure_optimizer(self, client_optimizer, model_parameters):
        # 2 3 5 7
        if client_optimizer is None:
            basic_optimizer = DeepSpeedCPUAdam(model_parameters, adamw_mode=True)
        else:
            assert isinstance(client_optimizer, torch.optim.Optimizer), "optimizer is not torch.optim.Optimizer"
            basic_optimizer = client_optimizer

        basic_optimizer.param_groups[:] = [pg for pg in basic_optimizer.param_groups if len(pg["params"]) != 0]

        self.basic_optimizer = basic_optimizer

        self.optimizer = self._configure_zero_optimizer(basic_optimizer)

    # # # lr_scheduler # # #
    def _scheduler_from_config(self, optimizer):
        return optimizer

    def _configure_lr_scheduler(self):
        lr_scheduler = self._scheduler_from_config(self.optimizer)
        self.lr_scheduler = lr_scheduler

    # # # inner interface # # #
    def _take_model_step(self):
        self.optimizer.step()

        self._global_grad_norm = self.optimizer._global_grad_norm
        report_progress = self.global_rank == 0 if self.global_rank else True
        overflow = self.optimizer.overflow

        self._step_applied = not overflow

        if report_progress and (self.global_steps + 1) % self.steps_per_print() == 0:
            self._report_progress(self.global_steps + 1)

        self.losses = 0.0
        self.global_steps += 1
        self.global_samples += self.train_batch_size()

    def _scale_loss_by_gas(self, prescaled_loss, eval_micro_batches=None):
        # 1
        scaling_factor = self.gradient_accumulation_steps() if eval_micro_batches is None else eval_micro_batches
        scaled_loss = prescaled_loss / scaling_factor
        return scaled_loss

    def allreduce_gradients(self):
        # 1
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
        report_progress = self.global_rank == 0 if self.global_rank else True
        # 有True也有False 与globalrank有关 1 个True，3个False
        self.tput_timer.stop(global_step=self.is_gradient_accumulation_boundary(), report_speed=report_progress)
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
