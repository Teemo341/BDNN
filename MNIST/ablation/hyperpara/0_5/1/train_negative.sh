#! /bin/bash
#SBATCH -J -loss
#SBATCH -o result/BDNN_negative_train.out               
#SBATCH -p compute                  
#SBATCH --qos=debug               
#SBATCH -N 1                
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=1       
#SBATCH -t 24:00:00 

python -u resnet_bayesian_mnist.py