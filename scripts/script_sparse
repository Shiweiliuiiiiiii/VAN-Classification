#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gpus=4
#SBATCH -t 2-23:59:00
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH -o dense_LK_with1x1.out

source /home/shiweil/miniconda3/etc/profile.d/conda.sh
source activate pt1.10_cuda11.3

module purge
module load 2021
module load CUDA/11.3.1
#module load PyTorch/1.10.0-foss-2021a-CUDA-11.3.1


MODEL=van_tiny # van_{tiny, small, base, large}
DROP_PATH=0.1 # drop path rates [0.1, 0.1, 0.1, 0.2] for [tiny, small, base, large]
CUDA_VISIBLE_DEVICES=0,1,2,3 bash distributed_train2.sh 4 /scratch-shared/shiwei/  \
	   --model $MODEL -b 256 --lr 1e-3 --drop-path $DROP_PATH -j 72 -u 2000


source deactivate
