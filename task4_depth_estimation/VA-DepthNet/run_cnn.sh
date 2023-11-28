#!/bin/bash

#SBATCH --job-name=YOON
#SBATCH --output=log.traintinycnn.log
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=64000
#SBATCH --mail-user=yoonjung.choi@sjsu.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --partition=gpu

echo ':: Start ::'
source ~/.bashrc
conda activate newDepth
python vadepthnet/traincnn.py configs/yoon_arguments_train_kittieigen_cnn.txt
echo ':: End ::'



