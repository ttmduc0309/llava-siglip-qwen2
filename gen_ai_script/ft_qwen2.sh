#!/bin/bash
# nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l
wandb online
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=20002 llava/train/train_mem.py \
    --model_name_or_path /workspace/Llava_Qwen2/models/qwen_1.5b \
    --version qwen_2 \
    --data_path /workspace/Llava_Qwen2/data/Vietnamese-LLaVA-Instruct-150K-gg-translated/train-first-turn.json \
    --image_folder /workspace/Llava_Qwen2/data/images/coco/train2017 \
    --vision_tower /workspace/Llava_Qwen2/models/lingual_siglip \
    --pretrain_mm_mlp_adapter /workspace/Llava_Qwen2/checkpoints/Qwen2-1.5B-pretrain-multilingual-siglip-cc/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/Qwen2-1.5B-Instruct-Vision-150k-first-turn \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
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
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
    # --deepspeed ./scripts/zero2.json \
