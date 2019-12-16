#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -t30:00:00
#SBATCH --mem=10GB
#SBATCH --mail-type=END
#SBATCH --mail-user=zl2521@nyu.edu
#SBATCH --job-name=test-add-dc
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%j.out

module purge
source activate IIC

nohup python -m code.scripts.cluster.cluster_sobel --dataset STL10 --dataset_root /scratch/zl2521/cv/fair-sslime/data --out_root /scratch/zl2521/IIC_py2/IIC/just_test_output --model_ind 653 --arch ClusterNet5g --num_epochs 100 --output_k 140 --gt_k 10 --lr 0.0001 --lamb 1.0 --num_sub_heads 5 --batch_sz 560 --num_dataloaders 2 --save_freq 3 --mix_train --crop_orig --rand_crop_sz 64 --input_sz 64 --mode IID+
