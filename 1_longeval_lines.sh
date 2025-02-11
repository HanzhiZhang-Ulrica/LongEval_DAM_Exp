#!/bin/bash

model_names=(
    # "HanzhiZhang_DAM_LLaMA_1B"
    # "HanzhiZhang_H2O_LLaMA_1B"
    # "HanzhiZhang_MoA_LLaMA_1B"
    # "HanzhiZhang_StreamingLLM_LLaMA_1B"
    # "meta-llama_Llama-3.2-1B-Instruct"

    "HanzhiZhang_DAM_LLaMA_3B"
    # "HanzhiZhang_H2O_LLaMA_3B"
    # "HanzhiZhang_MoA_LLaMA_3B"
    # "HanzhiZhang_StreamingLLM_LLaMA_3B"
    # "meta-llama_Llama-3.2-3B-Instruct"
)

for model_name in "${model_names[@]}"; do
    if [[ "$model_name" == *"DAM"* ]]; then
        echo "Launching with Accelerate for model: $model_name"
        accelerate launch longeval/eval.py \
            --model-name-or-path "model/$model_name" \
            --task lines \
            --count 5 \
            --max_gpu_memory 40 \
            --test_dir longeval/evaluation \
            --num_gpus 4 &
    else
        echo "Launching with python for model: $model_name"
        python3 longeval/eval.py \
            --model-name-or-path "model/$model_name" \
            --task lines \
            --count 2 \
            --max_gpu_memory 40 \
            --test_dir longeval/evaluation \
            --num_gpus 4 &
    fi
    wait
done
