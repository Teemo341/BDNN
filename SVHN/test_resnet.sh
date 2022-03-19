#! /bin/bash
#SBATCH -J resnet_test
#SBATCH -o result/resnet_test.out                
#SBATCH -p compute                  
#SBATCH --qos=debug               
#SBATCH -N 1                
#SBATCH --ntasks-per-node=1                    
#SBATCH --cpus-per-task=12      
#SBATCH -t 24:00:00 
 
     
#SBATCH -t 24:00:00 
python -u test_detection.py --pre_trained_net save_resnet_svhn/final_model --network resnet --dataset svhn --out_dataset cifar10 