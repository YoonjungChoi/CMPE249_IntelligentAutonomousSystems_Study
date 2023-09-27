#!/bin/bash

#SBATCH --job-name=YOON
#SBATCH --output=log.train.log
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
conda activate cmpe249
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
python MyYolov7Train.py --device 0 --batch-size 8 --epochs 100 --img 640 640 --data data/waymo_data.yaml --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/yolov7x-waymo.yaml --weights yolov7x.pt --name yolov7x-waymo-mytrain
echo ':: End ::'



