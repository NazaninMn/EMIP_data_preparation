#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=150gb
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -A gutintelligencelab
#SBATCH -o test_.out

module purge
module load apptainer
apptainer run --nv ~/pytorch-1.8.1.sif main.py 

