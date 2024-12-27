#!/bin/bash
# nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l
wandb online
python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=20003 llava/train/train_mem.py \
    --model_name_or_path /workspace/Llava_Qwen2/checkpoints/Qwen2-1.5B-Instruct-Vision-150k-image-token \
    --version qwen_2 \
    --data_path /workspace/Llava_Qwen2/data/VietDoc-merged-split-converted.json \
    --image_folder /workspace/Llava_Qwen2/data \
    --vision_tower /workspace/Llava_Qwen2/models/lingual_siglip \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/Qwen2-1.5B-Instruct-Vision-Instruct150k-VietDoc1-image-token-split \
    --num_train_epochs 0.4 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb
    # --deepspeed ./scripts/zero2.json \
