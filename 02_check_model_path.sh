#!/bin/bash

# Define the array of model names
model_names=(
    "model/HanzhiZhang_H2O_LLaMA_1B"
    "model/HanzhiZhang_StreamingLLM_LLaMA_1B"
    "model/HanzhiZhang_MoA_LLaMA_1B"
    "model/HanzhiZhang_DAM_LLaMA_1B"
    "model/meta-llama_Llama-3.2-1B-Instruct"
)

# Iterate over each model and check for config.json
for model in "${model_names[@]}"; do
    if [ -d "$model" ]; then
        if [ -f "$model/config.json" ]; then
            echo "✔️ config.json exists in '$model'."
        else
            echo "❌ config.json is missing in '$model'."
        fi
    else
        echo "❌ Directory '$model' does not exist."
    fi
done


