#!/bin/bash

python -m llava.serve.cli \
    --model-path /workspace/Llava_Qwen2/checkpoints/Qwen2-1.5B-Instruct-Vision-Instruct150k-VietDoc1-first-turn \
    --image-file /workspace/Llava_Qwen2/data/output/images/train-00000-of-00026.parquet_image_0.png \
    --conv_mode "qwen_2"
    # --image-file "https://llava-vl.github.io/static/images/view.jpg" \