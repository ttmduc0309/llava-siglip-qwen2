# LLaVA-Qwen2: Enhanced with Qwen2 Base Model

## Download model weight
`huggingface-cli download toongs/llava-qwen2-siglip-Vietdoc50k --local-dir models/llava-qwen2`


## Pretrain Qwen2

```bash
bash pretrain_qwen2.sh
```

The checkpoint folder contains the checkpoint for the pretrained model

## Finetune Qwen2

```bash
bash ft_qwen2.sh
```

## Interface

```bash
bash run_cli.sh
```

## Installation

This repository builds upon the original LLaVA project, integrating the Qwen2 base model for improved performance.

1. Clone this repository and navigate to the custom LLaVA folder

    ```bash
    git clone https://github.com/ttmduc0309/llava-siglip-qwen2.git
    cd llava-siglip-qwen2
    ```

2. Install Package

    ```shell
    conda create -n llava python=3.10 -y
    conda activate llava
    pip install --upgrade pip  # enable PEP 660 support
    pip install -e .
    ```

3. Install additional packages for training cases

    ```shell
    pip install -e ".[train]"
    pip install flash-attn --no-build-isolation
    ```
