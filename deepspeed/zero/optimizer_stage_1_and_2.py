import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from collections import OrderedDict
from deepspeed import comm as dist
import types
from dataclasses import dataclass
from typing import Dict
import os

from deepspeed.utils import logger
from deepspeed.utils.constants import *
from .loss_scaler import CreateLossScaler
from .utils import *

pg_correctness_test = False


class DeepSpeedZeroOptimizer(object):
    def __init__(self,
                 init_optimizer,
                 param_names,
                 static_loss_scale=1.0,
                 dynamic_loss_scale=False,
                 dynamic_loss_args=None,
                 verbose=True,
                 contiguous_gradients=True,
                 reduce_bucket_size=500000000,
                 use_multi_rank_bucket_allreduce=True,
                 allgather_bucket_size=5000000000,
                 dp_process_group=None,
                 expert_parallel_group=None,
                 expert_data_parallel_group=None,
                 reduce_scatter=True,
                 overlap_comm=False,
                 offload_optimizer_config=None,
                 mpu=None,
                 clip_grad=0.0,
                 gradient_accumulation_dtype=torch.float32,
                 communication_data_type=torch.float16,
                 postscale_gradients=True,
                 gradient_predivide_factor=1.0,
                 gradient_accumulation_steps=1,
                 ignore_unused_parameters=True,
                 partition_grads=True,
                 round_robin_gradients=False,
                 has_moe_layers=False,
                 fp16_master_weights_and_gradients=False,
                 elastic_checkpoint=False):

        self.cpu_offload = True
        self.cpu_offload_pin_memory = False
        self.elastic_checkpoint = elastic_checkpoint
        self.param_names = param_names
        self.mpu = mpu
        self.optimizer = init_optimizer
        self.flatten = _flatten_dense_tensors
        self.unflatten = _unflatten_dense_tensors
        self.partition_gradients = partition_grads
        self.zero_stage_string = "ZeRO-2"
        self.reduce_scatter = reduce_scatter
        self.overlap_comm = overlap_comm
        self.deepspeed_adam_offload = True
        self.device = 'cpu'
        self.dp_process_group = dp_process_group
        self.sequence_parallel_size = 1
        self.ep_process_group = expert_parallel_group
        self.expert_dp_process_group = expert_data_parallel_group
        dp_size = dist.get_world_size(group=self.dp_process_group)
        self.real_dp_process_group = [dp_process_group for i in range(len(self.optimizer.param_groups))]
        self.partition_count = [dp_size for i in range(len(self.optimizer.param_groups))]
        self.is_gradient_accumulation_boundary = True
        self.contiguous_gradients = contiguous_gradients or self.cpu_offload
        self.has_moe_layers = has_moe_layers
        self._global_grad_norm = 0.
        self.model_parallel_group = None
        self.model_parallel_world_size = 1
        self.model_parallel_rank = 0
        self.overflow = False
        self.clip_grad = clip_grad
        self.communication_data_type = communication_data_type
        self.gradient_predivide_factor = gradient_predivide_factor
        self.postscale_gradients = postscale_gradients
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.micro_step_id = 0
        self.ignore_unused_parameters = ignore_unused_parameters
        self.round_robin_gradients = round_robin_gradients
        self.extra_large_param_to_reduce = None
        self.fp16_master_weights_and_gradients = fp16_master_weights_and_gradients
        self.bit16_groups = []
        self.bit16_groups_flat = []
        self.parallel_partitioned_bit16_groups = []
        self.single_partition_of_fp32_groups = []
        self.params_not_in_partition = []
        self.params_in_partition = []
        self.first_offset = []
        self.partition_size = []
        self.nccl_start_alignment_factor = 2
        self.all_reduce_print = False
        self.dtype = self.optimizer.param_groups[0]['params'][0].dtype
        self.gradient_accumulation_dtype = gradient_accumulation_dtype
        self.use_separate_grad_accum = False
        self.use_grad_accum_attribute = False
        self.round_robin_bit16_groups = []
        self.round_robin_bit16_indices = []
        self.round_robin_bit16_meta = []
        self.groups_padding = []
        self.ipg_index = 0

        for i, param_group in enumerate(self.optimizer.param_groups):
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            trainable_parameters = []
            for param in param_group['params']:
                if param.requires_grad:
                    param.grad_accum = None
                    trainable_parameters.append(param)
            self.bit16_groups.append(trainable_parameters)
            orig_group_numel = 0
            for param in self.bit16_groups[i]:
                orig_group_numel += param.numel()
                param.cpu_data = param.data.cpu()
                param.data = torch.empty(1).to(param.device)

            move_to_cpu(self.bit16_groups[i])
            empty_cache()

            round_robin_tensors = self.bit16_groups[i]
            round_robin_indices = list(range(len(self.bit16_groups[i])))

            self.round_robin_bit16_groups.append(round_robin_tensors)
            self.round_robin_bit16_indices.append(round_robin_indices)
            meta_tensors = []
            for param in round_robin_tensors:
                meta_tensors.append(torch.zeros_like(param.cpu_data, device="meta"))
            self.round_robin_bit16_meta.append(meta_tensors)
            flattened_buffer = self.flatten_dense_tensors_aligned(
                self.round_robin_bit16_groups[i],
                self.nccl_start_alignment_factor * dist.get_world_size(group=self.real_dp_process_group[i]))
            for param in self.bit16_groups[i]:
                del param.cpu_data
            self.bit16_groups_flat.append(flattened_buffer.to('cuda:{}'.format(torch.cuda.current_device())))
            del flattened_buffer
            if partition_id == dist.get_world_size(group=self.real_dp_process_group[i]) - 1:
                padding = self.bit16_groups_flat[i].numel() - orig_group_numel
            else:
                padding = 0

            self.groups_padding.append(padding)

            self._update_model_bit16_weights(i)

            data_parallel_partitions = self.get_data_parallel_partitions(self.bit16_groups_flat[i], i)
            self.parallel_partitioned_bit16_groups.append(data_parallel_partitions)

            weights_partition = self.parallel_partitioned_bit16_groups[i][partition_id].to(
                self.device).clone().float().detach()

            weights_partition = weights_partition.pin_memory()

            self.single_partition_of_fp32_groups.append(weights_partition)
            self.single_partition_of_fp32_groups[i].requires_grad = True
            param_group['params'] = [self.single_partition_of_fp32_groups[i]]
            partition_size = len(self.bit16_groups_flat[i]) / dist.get_world_size(group=self.real_dp_process_group[i])
            params_in_partition, params_not_in_partition, first_offset = self.get_partition_info(
                self.round_robin_bit16_groups[i], partition_size, partition_id)
            self.partition_size.append(partition_size)
            self.params_in_partition.append(params_in_partition)
            self.params_not_in_partition.append(params_not_in_partition)
            self.first_offset.append(first_offset)
        self.reduce_bucket_size = int(reduce_bucket_size)
        self.use_multi_rank_bucket_allreduce = use_multi_rank_bucket_allreduce
        self.allgather_bucket_size = int(allgather_bucket_size)

        self.reduction_stream = torch.cuda.Stream
        self.callback_queued = False
        self.param_dict = {}
        self.is_param_in_current_partition = {}
        self.grads_in_ipg_bucket = []
        self.params_in_ipg_bucket = []
        self.elements_in_ipg_bucket = 0
        self.params_already_reduced = []
        self._release_ipg_buffers()
        self.previous_reduced_grads = None
        self.ipg_bucket_has_moe_params = False
        self.param_id = {}

        largest_param_numel = 0
        count = 0
        for i, params_group in enumerate(self.bit16_groups):
            for param in params_group:
                unique_id = id(param)
                self.param_id[unique_id] = count
                self.param_dict[count] = param
                self.params_already_reduced.append(False)
                if param.numel() > largest_param_numel:
                    largest_param_numel = param.numel()
                count = count + 1

        for param_group in self.params_in_partition:
            for param in param_group:
                self.is_param_in_current_partition[self.get_param_id(param)] = True

        for param_group in self.params_not_in_partition:
            for param in param_group:
                self.is_param_in_current_partition[self.get_param_id(param)] = False

        self.accumulated_grads_in_cpu = {}
        self.norm_for_param_grads = {}
        self.local_overflow = False
        self.grad_position = {}
        self.temp_grad_buffer_for_cpu_offload = torch.zeros(largest_param_numel,
                                                            device=self.device,
                                                            dtype=self.dtype)
        self.temp_grad_buffer_for_gpu_offload = torch.zeros(largest_param_numel,
                                                            device='cuda:{}'.format(torch.cuda.current_device()),
                                                            dtype=self.dtype)
        for i, params_group in enumerate(self.bit16_groups):
            self.get_grad_position(i, self.params_in_partition[i], self.first_offset[i], self.partition_size[i])

        self.param_to_partition_ids = {}
        self.is_partition_reduced = {}
        self.remaining_grads_in_partition = {}
        self.total_grads_in_partition = {}
        self.is_grad_computed = {}
        self.grad_partition_insertion_offset = {}
        self.grad_start_offset = {}
        self.averaged_gradients = {}
        self.offload_gradient_dict = {}
        self.first_param_index_in_partition = {}
        self.initialize_gradient_partitioning_data_structures()
        self.reset_partition_gradient_structures()
        self._grad_acc_hooks = []
        self.grad_accs = []
        self.create_reduce_and_remove_grad_hooks()
        self.custom_loss_scaler = False
        self.external_loss_scale = None
        self.loss_scaler = CreateLossScaler(dtype=self.dtype,
                                            static_loss_scale=static_loss_scale,
                                            dynamic_scaling=dynamic_loss_scale,
                                            dynamic_loss_args=dynamic_loss_args)
        self.dynamic_loss_scale = self.loss_scaler.dynamic

        self.initialize_optimizer_states()

        self._link_all_hp_params()
        self._hp_optimizer_states_linked = False

        self._enable_universal_checkpoint()
        self._param_slice_mappings = self._create_param_mapping()

    def _update_model_bit16_weights(self, group_index):
        updated_params = self.unflatten(self.bit16_groups_flat[group_index],
                                        self.round_robin_bit16_groups[group_index])
        for p, q in zip(self.round_robin_bit16_groups[group_index], updated_params):
            p.data = q.data

        # set model fp16 weight to slices of reordered flattened buffer
        for param_index, param in enumerate(self.bit16_groups[group_index]):
            new_index = self.round_robin_bit16_indices[group_index][param_index]
            # noinspection PyUnresolvedReferences
            param.data = self.round_robin_bit16_groups[group_index][new_index].data

    def get_data_parallel_partitions(self, tensor, group_id):
        partitions = []

        dp = dist.get_world_size(group=self.real_dp_process_group[group_id])
        # dp_id = dist.get_rank(group=self.real_dp_process_group[group_id])

        total_num_elements = tensor.numel()

        base_size = total_num_elements // dp
        remaining = total_num_elements % dp

        start = 0
        for ids in range(dp):
            partition_size = base_size
            if ids < remaining:
                partition_size = partition_size + 1
            partitions.append(tensor.narrow(0, start, partition_size))
            start = start + partition_size
        return partitions

    def get_partition_info(self, tensor_list, partition_size, partition_id):
        params_in_partition = []
        params_not_in_partition = []

        start_index = partition_size * partition_id
        end_index = partition_size * (partition_id + 1)

        current_index = 0
        first_offset = 0

        for tensor in tensor_list:
            tensor_size = tensor.numel()
            if start_index <= current_index < end_index:
                params_in_partition.append(tensor)
            elif current_index < start_index < (current_index + tensor_size):
                params_in_partition.append(tensor)
                first_offset = start_index - current_index
            else:
                params_not_in_partition.append(tensor)
            current_index = current_index + tensor_size
        return params_in_partition, params_not_in_partition, first_offset

    def _release_ipg_buffers(self):
        self.ipg_buffer = None
        self.grads_in_partition = None
        self.grads_in_partition_offset = 0

    def get_grad_position(self, group_id, tensor_list, first_offset, partition_size):
        current_offset = 0

        for i, tensor in enumerate(tensor_list):
            param_id = self.get_param_id(tensor)
            param_start_offset = 0

            num_elements = tensor.numel()

            if i == 0 and first_offset > 0:
                tensor_offset = first_offset
                num_elements = num_elements - tensor_offset
                param_start_offset = first_offset

            if num_elements > (partition_size - current_offset):
                num_elements = partition_size - current_offset

            self.grad_position[param_id] = [
                int(group_id), int(param_start_offset),
                int(current_offset), int(num_elements)
            ]
            current_offset += num_elements

    def initialize_gradient_partitioning_data_structures(self):

        for i, param_group in enumerate(self.round_robin_bit16_groups):
            total_partitions = dist.get_world_size(group=self.real_dp_process_group[i])

            self.param_to_partition_ids[i] = {}
            self.is_partition_reduced[i] = {}
            self.total_grads_in_partition[i] = {}
            self.remaining_grads_in_partition[i] = {}
            self.is_grad_computed[i] = {}
            self.grad_partition_insertion_offset[i] = {}
            self.grad_start_offset[i] = {}
            self.first_param_index_in_partition[i] = {}

            for partition_id in range(total_partitions):
                self.is_grad_computed[i][partition_id] = {}
                self.grad_partition_insertion_offset[i][partition_id] = {}
                self.grad_start_offset[i][partition_id] = {}
                self.total_grads_in_partition[i][partition_id] = 0
                self.initialize_gradient_partition(i, param_group, partition_id)
                self.is_partition_reduced[i][partition_id] = False
                self.first_param_index_in_partition[i][partition_id] = self.get_first_param_index(
                    i, param_group, partition_id)

    def initialize_gradient_partition(self, i, param_group, partition_id):

        def set_key_value_list(dictionary, key, value):
            if key in dictionary:
                dictionary[key].append(value)
            else:
                dictionary[key] = [value]

        def increment_value(dictionary, key):
            if key in dictionary:
                dictionary[key] += 1
            else:
                dictionary[key] = 1

        partition_size = self.partition_size[i]

        start_index = partition_size * partition_id
        end_index = partition_size * (partition_id + 1)

        current_index = 0
        first_offset = 0

        for param in param_group:

            param_size = param.numel()
            param_id = self.get_param_id(param)

            if start_index <= current_index < end_index:
                set_key_value_list(self.param_to_partition_ids[i], param_id, partition_id)
                increment_value(self.total_grads_in_partition[i], partition_id)

                self.is_grad_computed[i][partition_id][param_id] = False

                self.grad_partition_insertion_offset[i][partition_id][param_id] = current_index - start_index
                self.grad_start_offset[i][partition_id][param_id] = 0

            elif current_index < start_index < (current_index + param_size):
                first_offset = start_index - current_index

                set_key_value_list(self.param_to_partition_ids[i], param_id, partition_id)
                increment_value(self.total_grads_in_partition[i], partition_id)

                self.is_grad_computed[i][partition_id][param_id] = False

                self.grad_partition_insertion_offset[i][partition_id][param_id] = 0
                self.grad_start_offset[i][partition_id][param_id] = first_offset

            current_index = current_index + param_size

    def get_first_param_index(self, group_id, param_group, partition_id):
        for index, param in enumerate(param_group):
            param_id = self.get_param_id(param)
            if partition_id in self.param_to_partition_ids[group_id][param_id]:
                return index
        return None

    def get_param_id(self, param):
        unique_id = id(param)
        return self.param_id[unique_id]

    def reset_partition_gradient_structures(self):
        for i, _ in enumerate(self.bit16_groups):
            total_partitions = dist.get_world_size(group=self.real_dp_process_group[i])
            for partition_id in range(total_partitions):
                self.is_partition_reduced[i][partition_id] = False
                self.remaining_grads_in_partition[i][partition_id] = self.total_grads_in_partition[i][partition_id]

                for param_id in self.is_grad_computed[i][partition_id]:
                    self.is_grad_computed[i][partition_id][param_id] = False

    def create_reduce_and_remove_grad_hooks(self):
        for i, param_group in enumerate(self.bit16_groups):
            for param in param_group:
                if param.requires_grad:
                    def wrapper(param, i):
                        param_tmp = param.expand_as(param)
                        grad_acc = param_tmp.grad_fn.next_functions[0][0]

                        def reduce_partition_and_remove_grads(*notneeded):
                            self.reduce_ready_partitions_and_remove_grads(param, i)

                        self._grad_acc_hooks.append(grad_acc.register_hook(reduce_partition_and_remove_grads))
                        self.grad_accs.append(grad_acc)

                    wrapper(param, i)

    def reduce_ready_partitions_and_remove_grads(self, param, i):
        if self.partition_gradients or self.is_gradient_accumulation_boundary:
            self.reduce_independent_p_g_buckets_and_remove_grads(param, i)

    def reduce_independent_p_g_buckets_and_remove_grads(self, param, i):

        grad_reduc = self.get_gradient_for_reduction(param)

        param_id = self.get_param_id(param)

        new_grad_tensor = self.ipg_buffer[self.ipg_index].narrow(0, self.elements_in_ipg_bucket, param.numel())
        new_grad_tensor.copy_(grad_reduc.view(-1))
        grad_reduc.data = new_grad_tensor.data.view_as(grad_reduc)

        self.elements_in_ipg_bucket += param.numel()
        self.grads_in_ipg_bucket.append(grad_reduc)
        self.params_in_ipg_bucket.append((i, param, param_id))

    def get_gradient_for_reduction(self, param):
        if self.use_grad_accum_attribute:
            return param.grad_accum.to(self.dtype) if param.grad_accum is not None else None
        else:
            return param.grad

    def flatten_dense_tensors_aligned(self, tensor_list, alignment):
        return self.flatten(align_dense_tensors(tensor_list, alignment))

    def initialize_optimizer_states(self):

        for i, group in enumerate(self.bit16_groups):
            single_grad_partition = torch.zeros(int(self.partition_size[i]),
                                                dtype=self.single_partition_of_fp32_groups[i].dtype,
                                                device=self.device)
            self.single_partition_of_fp32_groups[i].grad = single_grad_partition
        if isinstance(self.optimizer, torch.optim.Adagrad):
            self.optimizer = torch.optim.Adagrad(self.single_partition_of_fp32_groups, **self.optimizer.defaults)
        else:
            self.optimizer.step()

        return

    def _get_offload_gradient_dict(self):
        for param_group_index, _ in enumerate(self.optimizer.param_groups):
            self.offload_gradient_dict[param_group_index] = []
            for lp_param in self.params_in_partition[param_group_index]:
                param_id = self.get_param_id(lp_param)
                [_, _, dest_offset, num_elements] = self.grad_position[param_id]
                dest_tensor = self.single_partition_of_fp32_groups[param_group_index].grad.view(-1).narrow(
                    0, dest_offset, num_elements)
                self.offload_gradient_dict[param_group_index].append(dest_tensor)

    def _link_all_hp_params(self):
        dp_world_size = dist.get_world_size(group=self.dp_process_group)
        self._get_offload_gradient_dict()

        for i, _ in enumerate(self.optimizer.param_groups):
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            partition_size = self.bit16_groups_flat[i].numel() // dp_world_size
            flat_hp_partition = self.single_partition_of_fp32_groups[i]
            link_hp_params(lp_param_list=self.bit16_groups[i],
                           flat_hp_partition=flat_hp_partition,
                           gradient_dict=self.averaged_gradients,
                           offload_gradient_dict=self.offload_gradient_dict,
                           use_offload=self.cpu_offload,
                           param_group_index=i,
                           partition_start=partition_id * partition_size,
                           partition_size=partition_size,
                           partition_optimizer_state=self.optimizer.state[flat_hp_partition],
                           dp_group=self.real_dp_process_group[i])

    def _enable_universal_checkpoint(self):
        for lp_param_group in self.bit16_groups:
            enable_universal_checkpoint(param_list=lp_param_group)

    def _create_param_mapping(self):
        param_mapping = []
        for i, _ in enumerate(self.optimizer.param_groups):
            param_mapping_per_group = OrderedDict()
            for lp in self.bit16_groups[i]:
                if lp._hp_mapping is not None:
                    lp_name = self.param_names[lp]
                    param_mapping_per_group[lp_name] = lp._hp_mapping.get_hp_fragment_address()
            param_mapping.append(param_mapping_per_group)

        return param_mapping

    def allreduce_bucket(self, bucket, rank=None, log=None, divide=True, process_group=None):
        rank = None
        tensor = self.flatten(bucket)

        process_group = self.dp_process_group if process_group is None else process_group

        tensor_to_allreduce = tensor

        if pg_correctness_test or self.sequence_parallel_size > 1:
            communication_data_type = torch.float32
        else:
            communication_data_type = self.communication_data_type

        if communication_data_type != tensor.dtype:
            tensor_to_allreduce = tensor.to(communication_data_type)

        if divide:
            tensor_to_allreduce.div_(dist.get_world_size(group=process_group) / float(self.sequence_parallel_size))

        if rank is None:
            #    "All Reducing"
            dist.all_reduce(tensor_to_allreduce, group=process_group)
        else:
            global_rank = dist.get_rank(process_group)
            dist.reduce(tensor_to_allreduce, global_rank, group=process_group)

        if communication_data_type != tensor.dtype and tensor is not tensor_to_allreduce:
            if rank is None or rank == dist.get_rank(group=process_group):
                tensor.copy_(tensor_to_allreduce)

        return tensor

    def allreduce_and_copy_with_multiple_ranks(self,
                                               small_bucket,
                                               log=None,
                                               divide=True,
                                               process_group=None,
                                               bucket_ranks=None):
        process_group = self.dp_process_group if process_group is None else process_group
        allreduced = self.allreduce_bucket(small_bucket, log=log, divide=divide, process_group=process_group)
        for buf, synced, bucket_rank in zip(small_bucket, self.unflatten(allreduced, small_bucket), bucket_ranks):
            if dist.get_rank(group=process_group) == bucket_rank:
                buf.copy_(synced)

    def allreduce_and_scatter(self, bucket, numel_per_bucket=500000000, log=None, divide=True, process_group=None):
        small_bucket = []
        small_bucket_ranks = []
        numel = 0
        allreduce_sizes = []

        for i, bucket_elem in enumerate(bucket):
            rank, tensor = bucket_elem
            small_bucket.append(tensor)
            small_bucket_ranks.append(rank)
            numel = numel + tensor.numel()
            if numel > numel_per_bucket:
                self.allreduce_and_copy_with_multiple_ranks(small_bucket,
                                                            log=None,
                                                            divide=divide,
                                                            process_group=process_group,
                                                            bucket_ranks=small_bucket_ranks)
                small_bucket = []
                small_bucket_ranks = []
                numel = 0

        if len(small_bucket) > 0:
            self.allreduce_and_copy_with_multiple_ranks(small_bucket,
                                                        log=None,
                                                        divide=divide,
                                                        process_group=process_group,
                                                        bucket_ranks=small_bucket_ranks)

    def average_tensor(self, tensor):
        stream = self.reduction_stream
        stream.wait_stream(torch.cuda.current_stream(None))
        with torch.cuda.stream(stream):
            rank_and_offsets = []
            real_dp_process_group = []
            curr_size = 0
            prev_id, prev_process_group = -1, None
            process_group = self.dp_process_group
            for i, param, param_id in self.params_in_ipg_bucket:
                process_group = self.dp_process_group
                grad_reduc = self.get_gradient_for_reduction(param)
                if self.ipg_bucket_has_moe_params:
                    process_group = self.expert_dp_process_group[param.group_name] if is_moe_param(
                        param) else self.dp_process_group
                    grad_reduc.data.div_(dist.get_world_size(group=process_group) / float(self.sequence_parallel_size))

                partition_ids = self.param_to_partition_ids[i][param_id]
                partition_size = self.partition_size[i]
                partition_ids_w_offsets = []
                for partition_id in partition_ids:
                    offset = self.grad_start_offset[i][partition_id][param_id]
                    partition_ids_w_offsets.append((partition_id, offset))
                partition_ids_w_offsets.sort(key=lambda t: t[1])
                for idx in range(len(partition_ids_w_offsets)):
                    partition_id, offset = partition_ids_w_offsets[idx]
                    if idx == len(partition_ids_w_offsets) - 1:
                        numel = param.numel() - offset
                    else:
                        numel = partition_ids_w_offsets[idx + 1][1] - offset

                    if partition_id == prev_id and process_group == prev_process_group:
                        prev_pid, prev_size, prev_numel = rank_and_offsets[-1]
                        rank_and_offsets[-1] = (prev_pid, prev_size, prev_numel + numel)
                    else:
                        rank_and_offsets.append((partition_id, curr_size, numel))
                        real_dp_process_group.append(process_group)
                    curr_size += numel
                    prev_id, prev_process_group = partition_id, process_group
            tensor.div_(dist.get_world_size(group=self.dp_process_group) / float(self.sequence_parallel_size))
            buckets = {}
            for i, (dst, bucket_offset, numel) in enumerate(rank_and_offsets):
                grad_slice = tensor.narrow(0, int(bucket_offset), int(numel))
                bucket_key = real_dp_process_group[i] if self.use_multi_rank_bucket_allreduce else (
                    dst, real_dp_process_group[i])
                if bucket_key not in buckets:
                    buckets[bucket_key] = []
                if self.use_multi_rank_bucket_allreduce:
                    buckets[bucket_key].append((dst, grad_slice))
                else:
                    buckets[bucket_key].append(grad_slice)

            for bucket_key in buckets:
                self.allreduce_and_scatter(buckets[bucket_key],
                                           numel_per_bucket=self.reduce_bucket_size,
                                           divide=self.ipg_bucket_has_moe_params,
                                           process_group=bucket_key)

    def clear_grad_attribute(self, param):
        if self.use_grad_accum_attribute:
            param.grad_accum = None
        else:
            param.grad = None

    def get_param_gradient_attribute(self, param):
        return param.grad_accum if self.use_grad_accum_attribute else param.grad

    def async_accumulate_grad_in_cpu_via_gpu(self, param):
        param_id = self.get_param_id(param)
        [i, source_offset, dest_offset, num_elements] = self.grad_position[param_id]

        dest_buffer = self.temp_grad_buffer_for_gpu_offload.view(-1).narrow(0, 0, param.numel())

        def buffer_to_accumulate_to_in_cpu():
            buffer = torch.zeros(param.numel(), dtype=param.dtype, device=self.device)
            return buffer.pin_memory() if self.cpu_offload_pin_memory else buffer

        def accumulate_gradients():
            grad_accum = self.get_param_gradient_attribute(param)
            dest_buffer.copy_(self.accumulated_grads_in_cpu[param_id].view(-1), non_blocking=True)
            grad_accum.data.view(-1).add_(dest_buffer)

        def copy_gradients_to_cpu():
            grad_accum = self.get_param_gradient_attribute(param)
            self.accumulated_grads_in_cpu[param_id].data.copy_(grad_accum.data.view(-1), non_blocking=True)

        if param_id not in self.accumulated_grads_in_cpu:
            self.accumulated_grads_in_cpu[param_id] = buffer_to_accumulate_to_in_cpu()

        if self.micro_step_id > 0:
            accumulate_gradients()

        if not self.is_gradient_accumulation_boundary:
            copy_gradients_to_cpu()

    def set_norm_for_param_grad_in_gpu(self, param):
        param_id = self.get_param_id(param)
        grad_accum = self.get_param_gradient_attribute(param)
        if grad_accum is None:
            accumulated_grad = param.grad
        else:
            accumulated_grad = grad_accum

        [i, source_offset, dest_offset, num_elements] = self.grad_position[param_id]

        start = source_offset
        accumulated_grad = accumulated_grad.view(-1).narrow(0, start, num_elements)

        self.norm_for_param_grads[param_id] = accumulated_grad.data.double().norm(2)

    @staticmethod
    def _has_inf_or_nan(x, j=None):
        try:
            cpu_sum = float(x.float().sum())
        except RuntimeError as instance:

            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
                return True
            return False

    def update_overflow_tracker_for_param_grad(self, param):
        grad_accum = self.get_param_gradient_attribute(param)
        if grad_accum is not None and self._has_inf_or_nan(grad_accum.data):
            self.local_overflow = True

    def async_inplace_copy_grad_to_fp32_buffer_from_gpu(self, param):
        param_id = self.get_param_id(param)

        [i, source_offset, dest_offset, num_elements] = self.grad_position[param_id]

        dest_tensor = self.single_partition_of_fp32_groups[i].grad.view(-1).narrow(0, dest_offset, num_elements)

        grad_accum = self.get_param_gradient_attribute(param)

        if grad_accum is None:
            src_tensor = grad_accum.view(-1).narrow(0, source_offset, num_elements)
        else:
            src_tensor = grad_accum.view(-1).narrow(0, source_offset, num_elements)
        if not self.fp16_master_weights_and_gradients:
            src_tensor = src_tensor.float()

        dest_tensor.copy_(src_tensor, non_blocking=True)
        param.grad = None

    def copy_grads_in_partition(self, param):
        self.async_accumulate_grad_in_cpu_via_gpu(param)

        if self.is_gradient_accumulation_boundary:
            self.set_norm_for_param_grad_in_gpu(param)
            self.update_overflow_tracker_for_param_grad(param)
            self.async_inplace_copy_grad_to_fp32_buffer_from_gpu(param)

    def reduce_ipg_grads(self):
        self.average_tensor(self.ipg_buffer[self.ipg_index].narrow(0, 0, self.elements_in_ipg_bucket))

        stream = self.reduction_stream

        with torch.cuda.stream(stream):
            for _, param, param_id in self.params_in_ipg_bucket:
                self.params_already_reduced[param_id] = True
                if not self.is_param_in_current_partition[param_id]:
                    self.clear_grad_attribute(param)
                elif self.contiguous_gradients:
                    self.copy_grads_in_partition(param)

        self.grads_in_ipg_bucket = []
        self.params_in_ipg_bucket = []
        self.ipg_bucket_has_moe_params = False
        self.elements_in_ipg_bucket = 0

    def _clear_previous_reduced_grads(self):
        if self.previous_reduced_grads is not None:
            for param in self.previous_reduced_grads:
                self.clear_grad_attribute(param)
            self.previous_reduced_grads = None

    def overlapping_partition_gradients_reduce_epilogue(self):
        self.reduce_ipg_grads()
        for i in range(len(self.params_already_reduced)):
            self.params_already_reduced[i] = False
        torch.cuda.synchronize(None)
        self._clear_previous_reduced_grads()
        self._release_ipg_buffers()
        self.zero_grad(set_to_none=True)

    def has_overflow_partitioned_grads_serial(self):
        for i in range(len(self.bit16_groups)):
            for j, grad in enumerate(self.averaged_gradients[i]):
                if grad is not None and self._has_inf_or_nan(grad.data, j):
                    return True
        return False

    def has_overflow_serial(self, params, is_grad_list=False):
        for p in params:
            if p.grad is not None and self._has_inf_or_nan(p.grad.data):
                return True

        return False

    def _model_parallel_all_reduce(self, tensor, op):
        if self.model_parallel_group is None or self.model_parallel_world_size == 1:
            pass
        else:
            dist.all_reduce(tensor=tensor, op=op, group=self.model_parallel_group)

    def has_overflow(self, partition_gradients=True):
        if partition_gradients:
            overflow = self.local_overflow if self.cpu_offload else self.has_overflow_partitioned_grads_serial()
            overflow_gpu = torch.cuda.ByteTensor([overflow])
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=self.dp_process_group)
        else:
            params = []
            for group in self.bit16_groups:
                for param in group:
                    params.append(param)

            overflow = self.has_overflow_serial(params, is_grad_list=partition_gradients)
            overflow_gpu = torch.cuda.ByteTensor([overflow])

        self._model_parallel_all_reduce(tensor=overflow_gpu, op=dist.ReduceOp.MAX)
        overflow = overflow_gpu[0].item()
        return bool(overflow)

    def _check_overflow(self, partition_gradients=True):
        self.overflow = self.has_overflow(partition_gradients)

    def check_overflow(self, partition_gradients=True):
        self._check_overflow(partition_gradients)

    def _get_loss_scale(self):
        return self.loss_scaler.cur_scale

    def _set_loss_scale(self, value):
        self.loss_scaler.cur_scale = value

    loss_scale = property(_get_loss_scale, _set_loss_scale)
    cur_scale = property(_get_loss_scale, _set_loss_scale)

    def _update_scale(self, has_overflow=False):
        self.loss_scaler.update_scale(has_overflow)

    def reset_cpu_buffers(self):
        self.norm_for_param_grads = {}
        self.local_overflow = False

    def complete_grad_norm_calculation_for_cpu_offload(self, params):
        total_norm = 0.0
        norm_type = 2.0
        for p in params:
            if hasattr(p, PIPE_REPLICATED) and p.ds_pipe_replicated:
                continue
            if is_model_parallel_parameter(p) or (self.model_parallel_rank == 0):
                param_id = self.get_param_id(p)
                # as some model have trainable parameters but skipped in training,
                # their backward hooks in self.create_reduce_and_remove_grad_hooks() will not run,
                # so they have no norm_for_param_grads
                if param_id in self.norm_for_param_grads:
                    param_norm = self.norm_for_param_grads[param_id]
                    total_norm += param_norm.item() ** 2

        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.SUM, group=self.dp_process_group)

        self._model_parallel_all_reduce(tensor=total_norm_cuda, op=dist.ReduceOp.SUM)

        total_norm = total_norm_cuda[0].item() ** (1. / norm_type)

        if total_norm == float('inf') or total_norm == -float('inf') or total_norm != total_norm:
            total_norm = -1

        return total_norm

    def scaled_global_norm(self, norm_type=2):
        norm_groups = []
        for i, group in enumerate(self.bit16_groups):
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            norm_groups.append(self.complete_grad_norm_calculation_for_cpu_offload(self.params_in_partition[i]))
            single_grad_partition = self.single_partition_of_fp32_groups[i].grad
        return get_global_norm(norm_list=norm_groups)

    def unscale_and_clip_grads(self, grad_groups_flat, total_norm):
        combined_scale = self.loss_scale
        if self.clip_grad > 0.:
            # norm is in fact norm*scale
            clip = ((total_norm / self.loss_scale) + 1e-6) / self.clip_grad
            if clip > 1:
                combined_scale = clip * self.loss_scale

        for grad in grad_groups_flat:
            if isinstance(grad, list):
                sub_partitions = grad
                for g in sub_partitions:
                    g.data.mul_(1. / combined_scale)
            else:
                grad.data.mul_(1. / combined_scale)

    def _optimizer_step(self, group_no):
        original_param_groups = self.optimizer.param_groups
        self.optimizer.param_groups = [original_param_groups[group_no]]
        self.optimizer.step()
        self.optimizer.param_groups = original_param_groups

    # #######################################
    def zero_grad(self, set_to_none=True):
        for group in self.bit16_groups:
            for p in group:
                if set_to_none:
                    p.grad = None  # epilogue and in step
                    p.grad_accum = None
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

    def backward(self, loss, retain_graph=False):
        self.micro_step_id += 1

        self.ipg_buffer = []
        buf_0 = torch.empty(int(self.reduce_bucket_size),
                            dtype=self.dtype,
                            device='cuda:{}'.format(torch.cuda.current_device()))
        self.ipg_buffer.append(buf_0)

        buf_1 = torch.empty(int(self.reduce_bucket_size),
                            dtype=self.dtype,
                            device='cuda:{}'.format(torch.cuda.current_device()))
        self.ipg_buffer.append(buf_1)
        self.ipg_index = 0

        self.loss_scaler.backward(loss.float(), retain_graph=retain_graph)

    def step(self, closure=None):
        self.micro_step_id = -1
        self.check_overflow()
        prev_scale = self.loss_scale
        self._update_scale(self.overflow)
        if self.overflow:
            self.zero_grad(set_to_none=True)
            if self.cpu_offload:
                self.reset_cpu_buffers()
            return

        scaled_global_grad_norm = self.scaled_global_norm()
        self._global_grad_norm = scaled_global_grad_norm / prev_scale
        for i, group in enumerate(self.bit16_groups):
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            single_grad_partition = self.single_partition_of_fp32_groups[i].grad
            self.unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)
            self._optimizer_step(i)
            bit16_partitions = self.parallel_partitioned_bit16_groups[i]
            fp32_partition = self.single_partition_of_fp32_groups[i]
            bit16_partitions[partition_id].data.copy_(
                fp32_partition.to('cuda:{}'.format(torch.cuda.current_device())).data)
        self.reset_cpu_buffers()

        all_gather_dp_groups(groups_flat=self.bit16_groups_flat,
                             partitioned_param_groups=self.parallel_partitioned_bit16_groups,
                             dp_process_group=self.real_dp_process_group,
                             start_alignment_factor=self.nccl_start_alignment_factor,
                             allgather_bucket_size=self.allgather_bucket_size)
        for i in range(len(self.bit16_groups)):
            self._update_model_bit16_weights(i)
        return
