#! /bin/bash
#SBATCH -J VBOE
#SBATCH -o result/VBOE_test_prob.out               
#SBATCH -p test                  
#SBATCH --qos=normal               
#SBATCH -N 1                
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=12      


python -u test_by_probability.py --pre_trained_net save_VBOE_mnist/final_model --network BBP --dataset mnist --out_dataset semeion