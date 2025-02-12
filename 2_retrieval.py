import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define directories
dataset_dir = "longeval/evaluation/lines/testcases/"
output_dir = "outputs/"

# List of models and their corresponding response directories
models = [
    ("LLaMA_3B", "LongEval/evaluation/lines_1/predictions/meta-llama_Llama-3.2-3B-Instruct/"),
    ("DAM_3B", "LongEval/evaluation/lines_1/predictions/HanzhiZhang_DAM_LLaMA_3B/"),
    ("MoA_3B", "LongEval/evaluation/lines_1/predictions/HanzhiZhang_MoA_LLaMA_3B/"),
    ("StreamingLLM_3B", "LongEval/evaluation/lines_1/predictions/HanzhiZhang_StreamingLLM_LLaMA_3B/"),
    ("H2O_3B", "LongEval/evaluation/lines_1/predictions/HanzhiZhang_H2O_LLaMA_3B/")
]

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Initialize a dictionary to store dataset info
dataset_info = {}

# Load dataset files
for dataset_filename in sorted(os.listdir(dataset_dir)):
    if dataset_filename.endswith("_lines.jsonl"):
        file_path = os.path.join(dataset_dir, dataset_filename)
        
        # Read JSONL file
        with open(file_path, "r") as file:
            for line in file:
                data = json.loads(line.strip())
                dataset_info[data["expected_number"]] = {
                    "num_lines": data["num_lines"],
                    "key_id": data["random_idx"][1],
                }

# Function to process each model's data
def process_model(model_name, response_dir):
    data = []
    
    for response_filename in sorted(os.listdir(response_dir)):
        if response_filename.endswith("_response.txt"):
            file_path = os.path.join(response_dir, response_filename)
            with open(file_path, "r") as file:
                lines = file.readlines()
            lines = lines[:-1]
            
            for line in lines:
                if "Label:" in line and "Parsed:" in line and "prompt length:" in line:
                    parts = line.split(",")
                    try:
                        label = int(parts[0].split(":")[1].strip())
                        parsed = int(parts[2].split(":")[1].strip())
                        prompt_length = int(parts[3].split(":")[1].strip())

                        if label in dataset_info:
                            num_lines = dataset_info[label]["num_lines"]
                            key_id = dataset_info[label]["key_id"]
                            retrieval_position = (key_id / num_lines) * 100
                            retrieval_position = max(0, min(retrieval_position, 100))
                            correct = 1 if label == parsed else 0
                            data.append([prompt_length, retrieval_position, correct])
                    except (IndexError, ValueError):
                        continue

    df = pd.DataFrame(data, columns=["prompt_length", "retrieval_position", "correct"])
    df["prompt_length_bin"] = (df["prompt_length"] // 1000) * 1000
    df["retrieval_position_bin"] = ((df["retrieval_position"] // 10) * 10).astype(int)
    accuracy_data = df.groupby(["prompt_length_bin", "retrieval_position_bin"])["correct"].mean().round(2).reset_index()
    accuracy_pivot = accuracy_data.pivot(index="retrieval_position_bin", columns="prompt_length_bin", values="correct")
    accuracy_pivot = accuracy_pivot.sort_index().sort_index(axis=1)
    accuracy_pivot = accuracy_pivot.reindex(sorted(accuracy_pivot.columns, key=int), axis=1)
    accuracy_pivot.columns = [f"{col//1000}k" for col in accuracy_pivot.columns]
    return accuracy_pivot

# Collect all models' accuracy data
data_dict = {model_name: process_model(model_name, response_dir) for model_name, response_dir in models}

# Get all possible prompt length bins across models
all_columns = sorted(set().union(*(df.columns for df in data_dict.values())), key=lambda x: int(x.replace("k", "")))

# Ensure all DataFrames have the same columns, filling missing values with NaN
data_dict = {model: df.reindex(columns=all_columns) for model, df in data_dict.items()}

# Generate heatmaps with a full-length color bar
fig, axes = plt.subplots(len(models), 1, figsize=(12, 1 * len(models) + 2), sharex=True, gridspec_kw={'hspace': 0})
cbar_ax = fig.add_axes([0.92, 0.145, 0.02,0.825])  # Extend colorbar to match the full height of the figure

# Draw heatmaps
for ax, (model_name, accuracy_pivot) in zip(axes, data_dict.items()):
    heatmap = sns.heatmap(
        accuracy_pivot,
        cmap="Set2",
        annot=False,
        linewidths=0.5,
        cbar=ax == axes[-1],  # Only add colorbar once
        cbar_ax=None if ax != axes[-1] else cbar_ax,
        square=True,
        ax=ax
    )
    ax.set_title("")
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=18, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10, fontweight='bold')

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 0.92, 1])  # Leave space for the colorbar on the right

# Save the figure
heatmap_path = os.path.join(output_dir, "retrieval_accuracy_heatmaps_with_full_colorbar.png")
plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)
plt.close()

print(f"- Combined Heatmap with Full-Length Color Bar Image: {heatmap_path}")
