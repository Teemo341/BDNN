#! /bin/bash
#SBATCH -J BDNN
#SBATCH -o result/BDNN_test_mat.out               
#SBATCH -p compute                  
#SBATCH --qos=debug               
#SBATCH -N 1                
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1   
#SBATCH -t 24:00:00 
#SBATCH -w node4


python -u test_by_matrix.py --pre_trained_net save_resnet_bayesian_mnist/final_model --network resnet_bayesian --dataset mnist --semi_out_dataset svhn --out_dataset cifar10