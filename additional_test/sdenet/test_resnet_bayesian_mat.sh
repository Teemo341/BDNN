#! /bin/bash
#SBATCH -J sdenet
#SBATCH -o result/sdenet_test_mat.out               
#SBATCH -p test                  
#SBATCH --qos=normal               
#SBATCH -N 1                
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=12       


python -u test_by_matrix.py --pre_trained_net save_sdenet_mnist/final_model --network sdenet --dataset cifar10_cat --semi_out_dataset cifar100_tiger --out_dataset mnist