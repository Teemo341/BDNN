#! /bin/bash
#SBATCH -J simple1
#SBATCH -o result/BDNN_simple_test_var.out               
#SBATCH -p compute                  
#SBATCH --qos=debug               
#SBATCH -N 1                
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=12       
#SBATCH -t 24:00:00 


python -u test_by_var.py --pre_trained_net save_resnet_bayesian_mnist/final_model --network resnet_bayesian --dataset mnist --out_dataset svhn 