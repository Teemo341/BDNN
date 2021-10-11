#! /bin/bash
#SBATCH -J BDNN
#SBATCH -o result/BDNN_train.out               
#SBATCH -p compute                  
#SBATCH --qos=debug               
#SBATCH -N 1                
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1   
#SBATCH -t 24:00:00 
#SBATCH -w node4

python -u resnet_bayesian_mnist.py