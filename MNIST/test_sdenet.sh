#! /bin/bash
#SBATCH -J sdenet_test
#SBATCH -o result/sdenet_test.out               
#SBATCH -p compute                  
#SBATCH --qos=debug               
#SBATCH -N 1                     
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=20  
#SBATCH --gres=gpu:1         
#SBATCH -t 24:00:00 

python test_detection.py --pre_trained_net save_sdenet_mnist/final_model --network sdenet --dataset mnist --out_dataset svhn