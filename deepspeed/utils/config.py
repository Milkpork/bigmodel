import time
import os
from datetime import timedelta

# global
Seed = 1234
# time_str = time.strftime('%Y%m%d%H%M%S', time.localtime())  # 全局time

# comm
Distributed_port = 29500  # 多机时的端口
Communication_backend_name = "nccl"  # dist初始化方法
Default_pg_timeout = timedelta(minutes=int(os.getenv("DEEPSPEED_TIMEOUT", default=30)))

# config
# train_batch_size = None
Train_micro_batch_size_per_gpu = 64
# gradient_accumulation_steps = 4
# steps_per_print = 10
# dump_state = False
Pin_memory = True

Zero_stage = 2  # now only support stage 2
Overlap_comm = False
Contiguous_gradients = True
Round_robin_gradients = False

Scheduler_name = "WarmupLR"
Scheduler_config = {
    "warmup_min_lr": 0,
    "warmup_max_lr": 0.001,
    "warmup_num_steps": 1000
}
