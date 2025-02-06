#!/bin/bash

python -m llava.serve.cli \
    --model-path /workspace/llava-siglip-qwen2/models/llava-qwen2 \
    --image-file /workspace/llava-siglip-qwen2/screen.png \
    --conv_mode "qwen_2"
    # --image-file "https://llava-vl.github.io/static/images/view.jpg" \