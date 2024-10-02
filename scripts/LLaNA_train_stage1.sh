master_port=$((RANDOM % (65535 - 49152 + 1) + 49152))
# Get the filename without extension
filename=$(basename "$0" | cut -f 1 -d '.')
datetime=$(date '+%d-%m-%Y_%H:%M')

model_name_or_path=outputs/LLaNA_7B_v1.1_init
root=data/llana_data/nerfllm
data_folder=shapenet_vecs
anno_folder=shapenet_texts
output_dir=outputs/LLaNA_7B_train_stage1/${filename}_${datetime}
point_backbone_ckpt=$model_name_or_path/point_bert_v1.2.pt

torchrun --nnodes=1 --nproc_per_node=3 --master_port=$master_port llana/train/train_mem_llana.py \
    --model_name_or_path $model_name_or_path \
    --root $root \
    --data_folder $data_folder \
    --anno_folder $anno_folder \
    --output_dir $output_dir \
    --version v1 \
    --model_max_length 2048 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy no \
    --save_strategy no \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --bf16 True \
    --fix_llm True \
    --gradient_checkpointing True \
    --report_to wandb \
    --run_name ${filename}