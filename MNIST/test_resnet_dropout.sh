#! /bin/bash
#SBATCH -J resnet_dropout_test
#SBATCH -o result/resnet_dropout_test.out               
#SBATCH -p compute                  
#SBATCH --qos=debug               
#SBATCH -N 1                     
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=20  
#SBATCH --gres=gpu:1         
#SBATCH -t 24:00:00 

python test_detection.py --pre_trained_net save_resnet_dropout_mnist/final_model --network mc_dropout --dataset mnist --out_dataset svhn