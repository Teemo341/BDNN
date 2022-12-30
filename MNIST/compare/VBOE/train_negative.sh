#! /bin/bash
#SBATCH -J VBOE
#SBATCH -o result/VBOE.out               
#SBATCH -p compute                  
#SBATCH --qos=normal               
#SBATCH -N 1                
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=12       
#SBATCH -t 24:00:00 

python -u VBOE_mnist.py