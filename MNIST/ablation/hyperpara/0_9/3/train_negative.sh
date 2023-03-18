#! /bin/bash
#SBATCH -J 0.9-3
#SBATCH -o result/BDNN_negative_train.out               
#SBATCH -p compute                  
#SBATCH --qos=normal               
#SBATCH -N 1                
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=1       
#SBATCH --gres=gpu:1

python -u resnet_bayesian_mnist.py