#!/bin/bash

python -m llava.serve.inference \
    --model-path /workspace/llava-siglip-qwen2/models/llava-qwen2 \
    --image-file /workspace/llava-siglip-qwen2/benchmark/30.jpg \
    --input-text "Cách mạng công nghiệp có mối liên hệ như thế nào với sự phát triển của vật lí thực nghiệm?" \
    --conv_mode "qwen_2" 
    # --image-file "https://llava-vl.github.io/static/images/view.jpg" \