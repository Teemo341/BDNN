#! /bin/bash
#SBATCH -J ssy
#SBATCH -o  result/sdenet_train.out               
#SBATCH -p compute                  
#SBATCH --qos=debug               
#SBATCH -N 1                     
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=12
#SBATCH -w node1_3060
     
#SBATCH -t 24:00:00 
python -u /home/ssy/BDNN/MNIST/resnet_mnist.py 