#!/bin/bash

python -m llava.serve.inference \
    --model-path /workspace/Llava_Qwen2/checkpoints/Qwen2-1.5B-Instruct-Vision-Instruct150k-VietDoc1-first-turn \
    --image-file /workspace/Llava_Qwen2/data/output2/images/train-00000-of-00034.parquet_image_0.png \
    --input-text "Đây là sách gì?" \
    --conv_mode "qwen_2" 
    # --image-file "https://llava-vl.github.io/static/images/view.jpg" \