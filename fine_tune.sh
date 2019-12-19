#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t10:00:00
#SBATCH --mem=20GB
#SBATCH --mail-type=END
#SBATCH --mail-user=zl2521@nyu.edu
#SBATCH --job-name=fine_tune_125_IIC
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%j.out

module purge
source activate IIC

nohup python -m code.scripts.semisup.IID_semisup_STL10  --model_ind 754 --old_model_ind 654 --out_root /scratch/zl2521/IIC_py2/IIC/IIC_dc_every_two_epoch_output --head_lr 0.001 --trunk_lr 0.0001 --arch SupHead5 --penultimate_features --random_affine --affine_p 0.5 --cutout --cutout_p 0.5 --cutout_max_box 0.7 --num_epochs 200 --new_batch_sz 100
