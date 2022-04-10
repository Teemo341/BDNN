#! /bin/bash
#SBATCH -J ABNN
#SBATCH -o result/ABNN_test_mat.out               
#SBATCH -p compute                  
#SBATCH --qos=debug               
#SBATCH -N 1                
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=12      
#SBATCH -t 24:00:00 


python -u test_by_matrix.py --pre_trained_net save_resnet_bayesian_mnist/final_model --network resnet_bayesian --dataset cifar10_cat --semi_out_dataset cifar100_tiger --out_dataset cifar100_bus