#!/bin/bash
#SBATCH --job-name=Pytorch-1819-early2
#SBATCH --output=/afm02/Q2/Q2067/Data/DeepLearningTestData/HPC/Output/usa/Pytorch-VIC-%j.out
#SBATCH --error=/afm02/Q2/Q2067/Data/DeepLearningTestData/HPC/Log/usa/Pytorch-VIC-%j.err
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

python /afm02/Q2/Q2067/Data/DeepLearningTestData/HPC/Project/vic/1819/hpc_classification_lstm_vic_evi.py