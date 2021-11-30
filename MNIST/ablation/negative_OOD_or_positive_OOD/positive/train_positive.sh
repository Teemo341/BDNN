#! /bin/bash
#SBATCH -J +loss
#SBATCH -o result/BDNN_positive_train.out               
#SBATCH -p compute                  
#SBATCH --qos=debug               
#SBATCH -N 1                
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=12       
#SBATCH -t 24:00:00 

python -u resnet_bayesian_mnist.py