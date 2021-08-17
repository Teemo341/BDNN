#! /bin/bash
#SBATCH -J ssy
#SBATCH -o 1.out               
#SBATCH -p compute                  
#SBATCH --qos=debug               
#SBATCH -N 1                     
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=20  
     
#SBATCH -t 24:00:00 
python -u /home/siu170066/SDE-Net-master/MNIST/resnet_mnist.py 