#!/bin/bash
#SBATCH --partition=math
#SBATCH --nodes=1
#SBATCH -c 64  #Numero di CPU per nodo
#SBATCH --time=2:00:00
#SBATCH --mem=300GB
#SBATCH --job-name ViT_pretrained
#SBATCH --output=/home/users/francesco.grienti.stud/MLNucleation/GRIENTI/PyTorchLearning/ComputerVision/MaskedAutoEncoder/plots/vit-pretrained/output.txt
#SBATCH --error=/home/users/francesco.grienti.stud/MLNucleation/GRIENTI/PyTorchLearning/ComputerVision/MaskedAutoEncoder/plots/vit-pretrained/error.txt
#SBATCH --account=mlnucleation
#SBATCH --gres=gpu:2  #N Gpus to use

source /exa/software/Spack-2023/spack/share/spack/setup-env.sh
spack load cuda@12.5.0 arch=linux-rocky8-icelake

EXP=$1
mkdir -p /home/users/francesco.grienti.stud/MLNucleation/GRIENTI/PyTorchLearning/ComputerVision/MaskedAutoEncoder/plots/vit-pretrained/$EXP

# shellcheck disable=SC2164
cd /home/users/francesco.grienti.stud/MLNucleation/GRIENTI/PyTorchLearning/ComputerVision/MaskedAutoEncoder/trainer
echo "Started ViT-pretrained evaluation..." 
python3 train_full.py --exp_name "$EXP"
echo "Training done!!"

