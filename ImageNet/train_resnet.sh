#! /bin/bash
#SBATCH -J resnet
#SBATCH -o  result/resnet_train.out               
#SBATCH -p compute1                 
#SBATCH --qos=normal              
#SBATCH -N 1                     
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
     

python -u /home/home_node4/ssy/BDNN/ImageNet/resnet_imagenet.py 