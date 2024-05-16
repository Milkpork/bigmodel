import torch
from deepspeed import comm as dist
import types
from dataclasses import dataclass
from typing import Dict
import os
from math import sqrt

from deepspeed.utils.constants import *


class fragment_address:
    def __init__(self, start, numel):
        self.start = start
        self.numel = numel


@dataclass
class tensor_fragment:
    lp_fragment: torch.Tensor
    lp_fragment_address: fragment_address
    hp_fragment: torch.Tensor
    hp_fragment_address: fragment_address
    optim_fragment: Dict
    gradient_dict: Dict
    offload_gradient_dict: Dict
    use_offload: bool
    param_group_index: int

    def update_hp(self):
        self.hp_fragment.data.copy_(self.lp_fragment.data)

    def update_lp(self):
        self.lp_fragment.data.copy_(self.hp_fragment.data)

    def get_optim_state_fragment(self, key):
        if key in self.optim_fragment:
            return self.optim_fragment[key]
        else:
            raise ValueError(f'{key} not found in optimizer state fragment')

    def get_hp_fragment_address(self):
        return self.hp_fragment_address

    def get_optim_state_keys(self):
        return list(self.optim_fragment.keys())

    def get_hp_fragment(self, optim_state_key=None):
        if optim_state_key is None:
            return self.hp_fragment
        return self.get_optim_state_fragment(optim_state_key)


def load_hp_checkpoint_state(self, folder, tp_rank, tp_world_size):
    hp_mapping = self._hp_mapping
    optim_state_keys = hp_mapping.get_optim_state_keys()
    hp_keys = [FP32_WEIGHT_KEY] + optim_state_keys
    checkpoint_files = {key: os.path.join(folder, f"{key}.pt") for key in hp_keys}
    for file in checkpoint_files.values():
        assert os.path.isfile(file), f'{file} is not a valid file'

    for key in hp_keys:
        ckpt_file = checkpoint_files[key]
        ckpt_dict = torch.load(ckpt_file)
        full_hp_param = ckpt_dict[PARAM]

        if full_hp_param.shape == self.shape:
            tp_rank = 0
            tp_world_size = 1

        is_vocab_tensor = ckpt_dict.get(VOCAB_TENSOR, False)
        if is_vocab_tensor:

            padded_target_vocab_size = self.shape[0] * tp_world_size
            assert padded_target_vocab_size >= full_hp_param.shape[0], \
                f'Vocab tensor padded size {padded_target_vocab_size} < loaded universal size {full_hp_param.shape[0]}'
            if padded_target_vocab_size > full_hp_param.shape[0]:
                padding_size = padded_target_vocab_size - full_hp_param.shape[0]
                full_hp_param = torch.nn.functional.pad(full_hp_param, (0, 0, 0, padding_size), "constant", 0)

        full_param_numel = full_hp_param.numel()
        tp_slice_numel = self.numel()
        dst_tensor = hp_mapping.hp_fragment if key == FP32_WEIGHT_KEY else hp_mapping.get_optim_state_fragment(key)

        chunk_dim = ckpt_dict.get(CAT_DIM, 0)
        n_sub_params = ckpt_dict.get(PARAM_N_SUB_PARAMS, 1)
        if n_sub_params > 1:
            sub_params = full_hp_param.chunk(n_sub_params, dim=chunk_dim)
            sub_params_tp_slice = [p.chunk(tp_world_size, dim=chunk_dim)[tp_rank] for p in sub_params]
            tp_hp_slice = torch.cat(sub_params_tp_slice, dim=chunk_dim)
        else:
            tp_hp_slice = full_hp_param.chunk(tp_world_size, chunk_dim)[tp_rank]

        tp_hp_slice = tp_hp_slice.flatten()

        lp_frag_address = hp_mapping.lp_fragment_address
        tp_hp_fragment = tp_hp_slice.narrow(0, lp_frag_address.start, lp_frag_address.numel)
        assert dst_tensor.numel() == lp_frag_address.numel, \
            f'Load checkpoint {key} dst_tensor numel {dst_tensor.numel()} != src numel {lp_frag_address.numel}'

        dst_tensor.data.copy_(tp_hp_fragment.data)


def move_to_cpu(tensor_list):
    for tensor in tensor_list:
        tensor.data = tensor.data.cpu()


def empty_cache():
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
        return torch.cuda.reset_peak_memory_stats(None)


def get_full_hp_param(self, optim_state_key=None):
    reduce_buffer = torch.zeros_like(self, dtype=torch.float32).flatten()
    if self._hp_mapping is not None:
        lp_frag_address = self._hp_mapping.lp_fragment_address
        reduce_fragment = torch.narrow(reduce_buffer, 0, lp_frag_address.start, lp_frag_address.numel)
        hp_fragment = self._hp_mapping.get_hp_fragment(optim_state_key)
        reduce_fragment.data.copy_(hp_fragment.data)
    dist.all_reduce(reduce_buffer, group=self._dp_group)
    return reduce_buffer.reshape_as(self)


def get_full_hp_grad(self):
    reduce_buffer = torch.zeros_like(self, dtype=torch.float32).flatten()
    if self._hp_mapping is not None:
        hp_mapping = self._hp_mapping

        if hp_mapping.use_offload:
            gradient_dict = hp_mapping.offload_gradient_dict
        else:
            gradient_dict = hp_mapping.gradient_dict

        if hp_mapping.param_group_index not in gradient_dict or gradient_dict[hp_mapping.param_group_index] is None:
            raise ValueError("Gradients are only available immediately after backward and before engine step")

        lp_grad_fragment = gradient_dict[hp_mapping.param_group_index][self._index_in_param_group]
        hp_grad_fragment = lp_grad_fragment.to(torch.float32).flatten()

        lp_frag_address = self._hp_mapping.lp_fragment_address
        reduce_fragment = torch.narrow(reduce_buffer, 0, lp_frag_address.start, lp_frag_address.numel)

        if self.view(-1).shape == hp_grad_fragment.shape:
            reduce_buffer.data.copy_(hp_grad_fragment.data)
        else:
            reduce_fragment.data.copy_(hp_grad_fragment.data)

    dist.all_reduce(reduce_buffer, group=self._dp_group)
    return reduce_buffer.reshape_as(self)


def set_full_hp_param(self, value, optim_state_key=None):
    if self._hp_mapping is not None:
        lp_frag_address = self._hp_mapping.lp_fragment_address
        value_fragment = torch.narrow(value.flatten(), 0, lp_frag_address.start, lp_frag_address.numel)
        hp_fragment = self._hp_mapping.get_hp_fragment(optim_state_key)
        hp_fragment.data.copy_(value_fragment.data)


def _init_lp_to_hp_mapping(lp_param_list, partition_start, partition_size, dp_group):
    current_offset = 0
    param_and_offset_list = []
    partition_end = partition_start + partition_size
    index_in_param_group = 0
    for i, lp_param in enumerate(lp_param_list):
        lp_param._hp_mapping = None
        lp_param._dp_group = dp_group
        lp_param.get_full_hp_param = types.MethodType(get_full_hp_param, lp_param)
        lp_param.get_full_hp_grad = types.MethodType(get_full_hp_grad, lp_param)
        lp_param.set_full_hp_param = types.MethodType(set_full_hp_param, lp_param)

        lp_param_end = current_offset + lp_param.numel()
        if current_offset < partition_end and lp_param_end > partition_start:
            param_and_offset_list.append((lp_param, current_offset))
            lp_param._index_in_param_group = index_in_param_group

            index_in_param_group += 1
        current_offset += lp_param.numel()

    return param_and_offset_list


def align_dense_tensors(tensor_list, alignment):
    num_elements = sum(t.numel() for t in tensor_list)
    remaining = num_elements % alignment

    if remaining:
        elements_to_add = alignment - remaining
        pad_tensor = torch.zeros(elements_to_add, device=tensor_list[0].device, dtype=tensor_list[0].dtype)
        padded_tensor_list = tensor_list + [pad_tensor]
    else:
        padded_tensor_list = tensor_list

    return padded_tensor_list


def get_hp_fragment_mapping(lp_param, lp_start, flat_hp_partition, gradient_dict, offload_gradient_dict, use_offload,
                            param_group_index, partition_start, partition_size, optimizer_state_dict):
    lp_end = lp_param.numel() + lp_start
    hp_start = partition_start
    hp_end = partition_start + partition_size

    fragment_start = max(lp_start, hp_start)
    fragment_end = min(lp_end, hp_end)
    assert fragment_start < fragment_end, \
        f'fragment start {fragment_start} should be < fragment_end {fragment_end}'

    fragment_numel = fragment_end - fragment_start
    hp_frag_address = fragment_address(start=fragment_start - hp_start, numel=fragment_numel)
    hp_fragment_tensor = flat_hp_partition.narrow(0, hp_frag_address.start, hp_frag_address.numel)
    optim_fragment = {
        key: value.narrow(0, hp_frag_address.start, hp_frag_address.numel)
        for key, value in optimizer_state_dict.items()
        if torch.is_tensor(value) and value.shape == flat_hp_partition.shape
    }

    lp_frag_address = fragment_address(start=fragment_start - lp_start, numel=fragment_numel)
    lp_fragment_tensor = lp_param.flatten().narrow(0, lp_frag_address.start, lp_frag_address.numel)

    return tensor_fragment(lp_fragment=lp_fragment_tensor,
                           lp_fragment_address=lp_frag_address,
                           hp_fragment=hp_fragment_tensor,
                           hp_fragment_address=hp_frag_address,
                           optim_fragment=optim_fragment,
                           gradient_dict=gradient_dict,
                           offload_gradient_dict=offload_gradient_dict,
                           use_offload=use_offload,
                           param_group_index=param_group_index)


def link_hp_params(lp_param_list, flat_hp_partition, gradient_dict, offload_gradient_dict, use_offload,
                   param_group_index, partition_start, partition_size, partition_optimizer_state, dp_group):
    local_lp_param_and_offset = _init_lp_to_hp_mapping(lp_param_list, partition_start, partition_size, dp_group)

    for lp_param, lp_start in local_lp_param_and_offset:
        lp_param._hp_mapping = get_hp_fragment_mapping(lp_param, lp_start, flat_hp_partition, gradient_dict,
                                                       offload_gradient_dict, use_offload, param_group_index,
                                                       partition_start, partition_size, partition_optimizer_state)


def enable_universal_checkpoint(param_list):
    for param in param_list:
        param.load_hp_checkpoint_state = types.MethodType(load_hp_checkpoint_state, param)


def is_moe_param(param: torch.Tensor) -> bool:
    if hasattr(param, "allreduce") and not param.allreduce:
        return True
    return False


def get_global_norm(norm_list):
    """ Compute total from a list of norms
    """
    total_norm = 0.0
    for norm in norm_list:
        total_norm += norm ** 2.0
    # logger.info(f'norm_list = {norm_list} global = {sqrt(total_norm)}')
    return sqrt(total_norm)


def all_gather_into_tensor_dp_groups(groups_flat, partitioned_param_groups, dp_process_group):
    for group_id, (group_flat, partitioned_params) in enumerate(zip(groups_flat, partitioned_param_groups)):
        partition_id = dist.get_rank(group=dp_process_group[group_id])
        dp_world_size = dist.get_world_size(group=dp_process_group[group_id])
        if dp_world_size == 1:
            # no groups share optimizer states
            # pipeline parallel with bf16 will default call this even if dp size = 1.
            continue
        dist.all_gather_into_tensor(group_flat, partitioned_params[partition_id], dp_process_group[group_id])


def all_gather_dp_groups(groups_flat, partitioned_param_groups, dp_process_group, start_alignment_factor,
                         allgather_bucket_size):
    if dist.has_all_gather_into_tensor():
        return all_gather_into_tensor_dp_groups(groups_flat, partitioned_param_groups, dp_process_group)

    for group_id, partitioned_params in enumerate(partitioned_param_groups):
        # Sequential AllGather Best of both worlds
        partition_id = dist.get_rank(group=dp_process_group[group_id])
        dp_world_size = dist.get_world_size(group=dp_process_group[group_id])

        if dp_world_size == 1:
            # no groups share optimizer states
            # pipeline parallel with bf16 will default call this even if dp size = 1.
            continue
        num_shards = max(1, partitioned_params[partition_id].numel() * dp_world_size // allgather_bucket_size)

        shard_size = partitioned_params[partition_id].numel() // num_shards

        # Enforce nccl/rccl alignment of start location of each shard
        shard_size = shard_size - (shard_size % start_alignment_factor)

        num_elements = shard_size

        assert shard_size * num_shards <= partitioned_params[partition_id].numel()

        for shard_id in range(num_shards):

            if shard_id == (num_shards - 1):
                num_elements = partitioned_params[partition_id].numel() - shard_id * shard_size

            shard_list = []
            for dp_id in range(dp_world_size):
                curr_shard = partitioned_params[dp_id].narrow(0, shard_id * shard_size, num_elements).detach()
                shard_list.append(curr_shard)

            dist.all_gather(shard_list, shard_list[partition_id], dp_process_group[group_id])


def is_model_parallel_parameter(p) -> bool:
    if hasattr(p, 'model_parallel') and p.model_parallel:
        return True

    if hasattr(p, 'tensor_model_parallel') and p.tensor_model_parallel:
        return True

    return False
