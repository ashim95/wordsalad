#!/bin/bash
#
#SBATCH --partition=soc-gpu-kp
##SBATCH --partition=notchpeak-gpu
##SBATCH --account=notchpeak-gpu
#SBATCH --account=soc-gpu-kp
##SBATCH --gres=gpu:v100:1
#SBATCH --gres=gpu:1

#SBATCH --job-name=mnli_bert_base_100_epochs
#SBATCH --output=/scratch/kingspeak/serial/u1266434/robustness/text_attack/transformers/slurms/outputs/mnli_bert_base_100_epochs

#SBATCH --ntasks=16
#SBATCH --time=5-00:00:00
##SBATCH --mem=32000



WORK_DIR=/scratch/kingspeak/serial/u1266434/robustness/text_attack/transformers/
cd $WORK_DIR
#export LD_LIBRARY_PATH=/home/utah/ashim/cuda-10.0/lib64
export LD_LIBRARY_PATH=/uufs/chpc.utah.edu/common/home/u1266434/cuda-10.0/lib64

#Activate Environment
source ../../env_attack/bin/activate

mkdir -p logs

export GLUE_DIR=data
export TASK_NAME=MNLI
#export CUDA_VISIBLE_DEVICES=5


python examples/text-classification/run_glue.py --model_name_or_path bert-base-uncased --task_name MNLI --do_train --do_eval --data_dir data/mnli/ --max_seq_length 256 --per_device_eval_batch_size=16 --per_device_train_batch_size=32 --learning_rate 2e-5 --num_train_epochs 100.0 --output_dir ../models/mnli/bert-base-epochs_100 --overwrite_output_dir --save_steps 40000 | tee logs/bert-base-epochs_100.txt
