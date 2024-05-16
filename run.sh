#!/usr/bin/env bash

# global env
export HF_ENDPOINT=https://hf-mirror.com
export NCCL_P2P_DISABLE=1
export PDSH_RCMD_TYPE=ssh
export CUDA_VISIBLE_DEVICES=0,1
# export NCCL_SOCKET_IFNAME=eth2

SCRIPT_PATH="$( cd "$(dirname "$0")" ; pwd -P )"
SCRIPT_NAME=$(basename "$0")

LOG_DATE="$(date +'%Y%m%d')"
LOG_FILE="logs_DDP/${SCRIPT_NAME}.log-${LOG_DATE}"  # log2 requirement

cd "${SCRIPT_PATH}"
mkdir -p logs_DDP

exec &>>"$LOG_FILE"  # 将输出到LOG_FILE
(
flock -n 99 || exit 1

date +'{{{ %Y%m%d %H%M%S'

python -m torch.distributed.run --master_port 28999 --nproc_per_node 2 main.py

date +'}}} %Y%m%d %H%M%S'
) 99>"${SCRIPT_NAME}.lock"