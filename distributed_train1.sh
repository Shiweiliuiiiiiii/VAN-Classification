#!/bin/bash
NUM_PROC=$1
shift
mp=$RANDOM
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=10001 --master_addr=127.0.0.1 --master_port=$mp train.py "$@"

