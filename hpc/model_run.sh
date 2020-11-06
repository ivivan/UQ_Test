#!/bin/bash
#SBATCH --job-name=Pytorch-Moree
#SBATCH --output=/clusterdata/uqyzha77/Output/vic/Pytorch-Moree-%j.out
#SBATCH --error=/clusterdata/uqyzha77/Log/vic/big/full/Pytorch-Moree-%j.err
#SBATCH --mail-user=yifan.zhang@uq.edu.au
#SBATCH --mail-type=ALL

#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40G


module load cuda/11.0.2.450
module load gnu8/8.4.0
module load mvapich2-gnu4/2.3
# module load anaconda/3.7
# conda init bash
# source /opt/ohpc/pub/apps/anaconda2020/etc/profile.d/conda.sh
# conda activate /clusterdata/uqyzha77/.conda/envs/pytorch

python /clusterdata/uqyzha77/Project/vic/hpc_classification_lstm_single_full2.py