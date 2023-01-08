#! /bin/bash
#SBATCH -J OE1
#SBATCH -o result/OE_train.out               
#SBATCH -p compute                  
#SBATCH --qos=normal               
#SBATCH -N 1                
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=12       
#SBATCH -t 24:00:00 

python -u OE_mnist.py