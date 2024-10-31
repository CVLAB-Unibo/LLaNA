#!/bin/bash
#SBATCH -p boost_usr_prod
#SBATCH -n 4
#SBATCH --gres=gpu:4
#SBATCH -t 24:00:00
#SBATCH -A IscrC_V2Text 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --error=llana_objanerf_stage2.err
#SBATCH --output=llana_objanerf_stage2.out
#SBATCH -J llana

# Print SLURM environment variables for debugging
echo "NODELIST=${SLURM_NODELIST}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "SLURM_NTASKS=${SLURM_NTASKS}"
echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES}"

# Set MASTER_ADDR and MASTER_PORT
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=11111
export NCCL_NET=IB

# load coding environment
cd $FAST
source scratch_venv/bin/activate
export PATH="$PATH:/leonardo_scratch/fast/IscrC_V2Text/scratch_venv/bin"
module load cuda/12.1
cd dev/LLaNA/

master_port=$((RANDOM % (65535 - 49152 + 1) + 49152))
# Get the filename without extension
filename=$(basename "$0" | cut -f 1 -d '.')
datetime=$(date '+%d-%m-%Y_%H:%M')

#model_name_or_path=outputs/LLaNA_7b_train_stage1/LLaNA_7b_train_stage1    # set path to folder with stage 1 training results
model_name_or_path=/leonardo_scratch/fast/IscrC_V2Text/dev/LLaNA/outputs/LLaNA_7B_train_stage1_objanerf/slurm_script_30-10-2024_21:55
root=data/objanerf_text
data_folder=vecs
anno_folder=texts
output_dir=outputs/LLaNA_7B_train_stage2_objanerf/${filename}_${datetime}
export WANDB_MODE=offline

torchrun --nnodes=1 --nproc_per_node=4 --master_port=$master_port llana/train/train_mem_llana.py \
    --model_name_or_path $model_name_or_path \
    --root $root \
    --data_folder $data_folder \
    --anno_folder $anno_folder \
    --output_dir $output_dir \
    --version v1 \
    --model_max_length 2048 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy no \
    --eval_steps 100 \
    --save_strategy no \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --bf16 True \
    --fix_llm False \
    --gradient_checkpointing True \
    --stage_2 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --conversation_types "detailed_description" "single_round" "multi_round" \
    --report_to wandb \
    --run_name ${filename}