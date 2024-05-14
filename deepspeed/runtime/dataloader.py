import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from math import ceil

from deepspeed.utils.config import *


class DeepSpeedDataLoader(object):

    def __init__(self,
                 dataset,
                 batch_size,
                 local_rank):
        if local_rank >= 0:
            data_sampler = DistributedSampler(dataset=dataset, seed=Seed)
            device_count = 1
        else:
            data_sampler = RandomSampler(dataset)
            device_count = torch.cuda.device_count()
            batch_size *= device_count

        num_local_io_workers = 2 * device_count

        self.num_local_io_workers = num_local_io_workers
        self.data_sampler = data_sampler
        self.dataset = dataset
        self.device_count = device_count
        self.batch_size = batch_size
        self.pin_memory = Pin_memory
        self.data = None
        self.post_process_func = None
        self.dataloader_drop_last = False

        self.len = ceil(len(self.data_sampler) / self.batch_size)

    def __iter__(self):
        self._create_dataloader()
        return self

    def __len__(self):
        return self.len

    def __next__(self):
        return next(self.data)

    def _create_dataloader(self):
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.batch_size,
                                     pin_memory=self.pin_memory,
                                     sampler=self.data_sampler,
                                     num_workers=self.num_local_io_workers,
                                     drop_last=self.dataloader_drop_last)
        self.data = (x for x in self.dataloader)
        return self.dataloader
