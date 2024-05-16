import torch
from deepspeed import comm as dist
from deepspeed.utils import logger


class LossScalerBase:

    def __init__(self, cur_scale):
        self.cur_scale = cur_scale
        self.dynamic = False

    @property
    def loss_scale(self):
        return self.cur_scale

    def scale_gradient(self, module, grad_in, grad_out):
        return tuple(self.loss_scale * g for g in grad_in)

    def update_scale(self, overflow):
        pass

    def backward(self, loss, retain_graph=False):
        scaled_loss = loss * self.loss_scale
        scaled_loss.backward(retain_graph=retain_graph)


class DynamicLossScaler(LossScalerBase):

    def __init__(self,
                 init_scale=2 ** 32,
                 scale_factor=2.,
                 scale_window=1000,
                 min_scale=1,
                 delayed_shift=1,
                 consecutive_hysteresis=False,
                 raise_error_at_min_scale=True,
                 dtype=torch.half):
        super(DynamicLossScaler, self).__init__(init_scale)
        self.cur_iter = 0
        self.last_overflow_iter = -1
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.min_scale = min_scale
        self.delayed_shift = delayed_shift
        self.cur_hysteresis = delayed_shift
        self.consecutive_hysteresis = consecutive_hysteresis
        self.raise_error_at_min_scale = raise_error_at_min_scale
        self.dynamic = True
        self.dtype = dtype

    # `params` is a list / generator of torch.Variable
    def has_overflow_serial(self, params):
        for p in params:
            if p.grad is not None and self._has_inf_or_nan(p.grad.data):
                return True

        return False

    # `x` is a torch.Tensor
    def _has_inf_or_nan(self, x):
        try:
            # if x is half, the .float() incurs an additional deep copy, but it's necessary if
            # Pytorch's .sum() creates a one-element tensor of the same type as x
            # (which is true for some recent version of pytorch).
            cpu_sum = float(x.float().sum())
            # More efficient version that can be used if .sum() returns a Python scalar
            # cpu_sum = float(x.sum())
        except RuntimeError as instance:
            # We want to check if inst is actually an overflow exception.
            # RuntimeError could come from a different error.
            # If so, we still want the exception to propagate.
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum in [float('inf'), -float('inf')] or cpu_sum != cpu_sum:
                return True
            return False

    # `overflow` is boolean indicating whether the gradient overflowed
    def update_scale(self, overflow):
        if overflow:
            # self.cur_scale /= self.scale_factor
            if self.delayed_shift == 1 or self.cur_hysteresis == 1:
                if (self.cur_scale == self.min_scale) and self.raise_error_at_min_scale:
                    raise Exception(
                        "Current loss scale already at minimum - cannot decrease scale anymore. Exiting run.")
                else:
                    next_scale = max(self.cur_scale / self.scale_factor, self.min_scale)
                    if dist.get_rank() == 0:
                        overflow_msg = f"[deepspeed] OVERFLOW! Rank {dist.get_rank()} Skipping step."
                        if self.dtype == torch.half:
                            overflow_msg += f" Attempted loss scale: {int(self.cur_scale)}, reducing to {int(next_scale)}"
                        logger.info(overflow_msg)
                    self.cur_scale = next_scale
            else:
                if dist.get_rank() == 0:
                    overflow_msg = f"[deepspeed] OVERFLOW! Rank {dist.get_rank()} Skipping step."
                    if self.dtype == torch.half:
                        overflow_msg += f" Attempted loss scale: {int(self.cur_scale)}, but hysteresis is {self.cur_hysteresis}. Reducing hysteresis to {self.cur_hysteresis - 1}"
                    logger.info(overflow_msg)
                self.cur_hysteresis -= 1
            self.last_overflow_iter = self.cur_iter
        else:
            if self.consecutive_hysteresis:
                if dist.get_rank() == 0:
                    hysteresis_msg = f"Consecutive hysteresis is enabled. Restoring hysteresis to {self.delayed_shift}"
                    logger.info(hysteresis_msg)
                self.cur_hysteresis = self.delayed_shift
            if (self.cur_iter - self.last_overflow_iter) % self.scale_window == 0:
                if not self.consecutive_hysteresis:
                    self.cur_hysteresis = self.delayed_shift
                self.cur_scale *= self.scale_factor
        self.cur_iter += 1


def CreateLossScaler(dtype, static_loss_scale, dynamic_scaling, dynamic_loss_args):
    return DynamicLossScaler(dtype=dtype)
