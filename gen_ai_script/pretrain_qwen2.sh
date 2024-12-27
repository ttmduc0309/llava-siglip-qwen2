#!/bin/bash
wandb online
export CUDA_VISIBLE_DEVICES=0,1,2,3
# python -m torch.distributed.run --nnodes=1 --nproc_per_node=3 --master_port=20001 llava/train/train_mem.py \
deepspeed --include=localhost:0,1,2,3 llava/train/train_mem.py \
    --model_name_or_path /workspace/Llava_Qwen2/models/qwen_1.5b \
    --version plain \
    --data_path /workspace/Llava_Qwen2/data/Vietnamese-liuhaotian-LLaVA-Pretrain-gg-translated/valid_entries.json \
    --image_folder /workspace/Llava_Qwen2/data/LLaVA-Pretrain/images \
    --vision_tower /workspace/Llava_Qwen2/models/lingual_siglip \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/Qwen2-1.5B-pretrain-multilingual-siglip-cc \
    --num_train_epochs 1 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 32 \
    --lazy_preprocess True \
    --report_to wandb \
    --deepspeed /workspace/Llava_Qwen2/scripts/deepspeed/zero2.json 
