#!/bin/bash
NUM_PROC=$1
shift
mp=$RANDOM
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=$mp train.py "$@"

