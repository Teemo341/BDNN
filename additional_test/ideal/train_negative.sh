#! /bin/bash
#SBATCH -J plot
#SBATCH -o result/feedback         
#SBATCH -p test                  
#SBATCH --qos=normal               
#SBATCH -N 1                
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=12       

python -u plot.py