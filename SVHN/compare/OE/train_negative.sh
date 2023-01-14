#! /bin/bash
#SBATCH -J OE1
#SBATCH -o result/OE_train.out               
#SBATCH -p test                  
#SBATCH --qos=normal               
#SBATCH -N 1                
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=12       

python -u OE_mnist.py