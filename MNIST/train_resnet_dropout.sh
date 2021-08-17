#! /bin/bash
#SBATCH -J resnet_dropout
#SBATCH -o result/resnet_dropout_train.out               
#SBATCH -p compute                  
#SBATCH --qos=debug               
#SBATCH -N 1                     
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=20  
#SBATCH --gres=gpu:1         
#SBATCH -t 24:00:00 
python -u  resnet_dropout_mnist.py