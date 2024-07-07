#! /bin/bash
#SBATCH -J WOODS
#SBATCH -o result/WOODS_train.out               
#SBATCH -p test                  
#SBATCH --qos=normal               
#SBATCH -N 1                
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=12       

python -u WOODS_mnist.py