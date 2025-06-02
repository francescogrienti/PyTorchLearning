#!/bin/bash
#SBATCH --partition=math
#SBATCH --nodes=1
#SBATCH -c 64  #Numero di CPU per nodo
#SBATCH --time=120:00:00
#SBATCH --mem=300GB
#SBATCH --job-name ViT_scratch
#SBATCH --output=/home/users/francesco.grienti.stud/MLNucleation/GRIENTI/PyTorchLearning/ComputerVision/MaskedAutoEncoder/plots/vit-scratch/output.txt
#SBATCH --error=/home/users/francesco.grienti.stud/MLNucleation/GRIENTI/PyTorchLearning/ComputerVision/MaskedAutoEncoder/plots/vit-scratch/error.txt
#SBATCH --account=mlnucleation
#SBATCH --gres=gpu:2  #N Gpus to use

source /exa/software/Spack-2023/spack/share/spack/setup-env.sh
spack load cuda@12.5.0 arch=linux-rocky8-icelake

EXP=$1
mkdir -p /home/users/francesco.grienti.stud/MLNucleation/GRIENTI/PyTorchLearning/ComputerVision/MaskedAutoEncoder/plots/vit-scratch/$EXP

cd /home/users/francesco.grienti.stud/MLNucleation/GRIENTI/PyTorchLearning/ComputerVision/MaskedAutoEncoder/trainer
echo "Started ViT-scratch evaluation..." 
python3 train_classifier.py --exp_name "$EXP"
echo "Training done!!"

