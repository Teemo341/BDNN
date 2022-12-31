#! /bin/bash
#SBATCH -J WOODS
#SBATCH -o result/WOODS_test_mat.out               
#SBATCH -p test                  
#SBATCH --qos=normal               
#SBATCH -N 1                
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=12      


python -u test_by_matrix.py --pre_trained_net save_resnet_WOODS_mnist/final_model --network resnet --dataset mnist --semi_out_dataset svhn --out_dataset cifar10