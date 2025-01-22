#!/bin/bash

model_names=(
    "HanzhiZhang_H2O_LLaMA_1B"
    "HanzhiZhang_StreamingLLM_LLaMA_1B"
    "HanzhiZhang_MoA_LLaMA_1B"
    "HanzhiZhang_DAM_LLaMA_1B"
    "meta-llama_Llama-3.2-1B-Instruct"
)

for model_name in "${model_names[@]}"; do
    python3 longeval/eval.py \
        --model-name-or-path "model/$model_name" \
        --task lines \
        --count 2 \
        --max_gpu_memory 40 \
        --test_dir longeval/evaluation \
        --num_gpus 4 &
    wait
done