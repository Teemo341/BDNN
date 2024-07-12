#! /bin/bash
#SBATCH -J swag
#SBATCH -o  result/swag_train_.out   

#SBATCH --partition = compute1   
#SBATCH --account   = compute1     
#SBATCH --qos       = compute1  
                          
#SBATCH --ntasks-per-node=1  
#SBATCH -N 1                        
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4090:1
     

python -u /home/home_node4/ssy/BDNN/ImageNet/swag_imagenet.py 