import os
import re
import matplotlib.pyplot as plt
import numpy as np
import itertools
from matplotlib.ticker import FuncFormatter

tpe = "3B"
iterations = ["1", "2", "3", "4", "5"]

# List of model names
model_names = [
    f"meta-llama_Llama-3.2-{tpe}-Instruct",
    f"HanzhiZhang_DAM_LLaMA_{tpe}",
    f"HanzhiZhang_H2O_LLaMA_{tpe}",
    f"HanzhiZhang_MoA_LLaMA_{tpe}",
    f"HanzhiZhang_StreamingLLM_LLaMA_{tpe}"
]

# Function to extract accuracy and prompt length
def extract_values(line):
    match = re.search(r"Accuracy: ([0-9\.]+), Ave Prompt Length: ([0-9\.]+)", line)
    if match:
        accuracy = float(match.group(1))
        length = float(match.group(2))
        return accuracy, length
    return None, None

# Collect data from all iterations
all_results = {model: {} for model in model_names}
lengths = {}

for ite in iterations:
    for model_name in model_names:
        folder_path = f"LongEval/evaluation/lines_{ite}/predictions/{model_name}/"
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(".txt"):
                line_number = int(re.search(r"(\d+)_response", filename).group(1))
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        accuracy, length = extract_values(last_line)
                        if accuracy is None or length is None:
                            accuracy = 0.0
                            length = 0.0
                        
                        if line_number not in all_results[model_name]:
                            all_results[model_name][line_number] = []
                        all_results[model_name][line_number].append(accuracy)
                        
                        # Store the length for each line number
                        if line_number not in lengths:
                            lengths[line_number] = length

# Filter and prepare data
line_numbers = sorted(set(itertools.chain.from_iterable(all_results[model].keys() for model in model_names)))
filtered_lines = [ln for ln in line_numbers if ln <= 3100]

# Prepare data for box plot and line plot
box_data = {model: [] for model in model_names}
avg_data = {model: [] for model in model_names}

for line_number in filtered_lines:
    for model in model_names:
        if line_number in all_results[model]:
            box_data[model].append(all_results[model][line_number])
            avg_data[model].append(np.mean(all_results[model][line_number]))
        else:
            box_data[model].append([])
            avg_data[model].append(0.0)

# Define colors with transparency (alpha)
colors = {
    f'meta-llama_Llama-3.2-{tpe}-Instruct': (0, 0, 0, 0.3),        # Black with transparency
    f'HanzhiZhang_DAM_LLaMA_{tpe}': (1, 0, 0, 0.3),               # Red with transparency
    f'HanzhiZhang_H2O_LLaMA_{tpe}': (0.5, 0, 0.5, 0.3),           # Purple with transparency
    f'HanzhiZhang_MoA_LLaMA_{tpe}': (0, 0, 1, 0.3),               # Blue with transparency
    f'HanzhiZhang_StreamingLLM_LLaMA_{tpe}': (0, 1, 0, 0.3)       # Green with transparency
}

# Define a mapping from model identifiers to display names
model_display_names = {
    f'meta-llama_Llama-3.2-{tpe}-Instruct': 'LLaMA 3.2 3B',
    f'HanzhiZhang_DAM_LLaMA_{tpe}': 'DAM',
    f'HanzhiZhang_H2O_LLaMA_{tpe}': 'H2O',
    f'HanzhiZhang_MoA_LLaMA_{tpe}': 'MoA',
    f'HanzhiZhang_StreamingLLM_LLaMA_{tpe}': 'StreamingLLM'
}

# Plot box plots and average accuracy lines
fig, ax = plt.subplots(figsize=(12, 8))
positions = np.array(filtered_lines)
linestyles = ['solid', 'dotted', 'dotted', 'dotted', 'dotted']

for i, (model, linestyle) in enumerate(zip(model_names, linestyles)):
    color = colors[model][:3]  # Extract RGB without alpha for lines
    box_color = colors[model]  # RGBA for box facecolor
    
    boxprops = dict(facecolor=box_color, edgecolor=color)
    whiskerprops = dict(color=color)
    capprops = dict(color=color)
    flierprops = dict(color=color, markeredgecolor=color)
    medianprops = dict(color=color)
    
    ax.boxplot(
        box_data[model], 
        positions=positions + i * 5, 
        widths=4, 
        patch_artist=True, 
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        flierprops=flierprops,
        medianprops=medianprops
    )
    ax.plot(
        positions + i * 5, 
        avg_data[model], 
        label=model_display_names[model],  # Use the display name here
        color=color, 
        linewidth=2, 
        linestyle=linestyle
    )

# Define the formatter function
def kilounit_formatter(x, pos):
    # Find the closest line number
    closest_line = min(filtered_lines, key=lambda ln: abs(ln - x))
    # Get the corresponding length
    length = lengths.get(closest_line, 0)
    return f'{length/1000:.1f}k'

# Set x-ticks at desired positions
ax.set_xticks(positions)

# Apply the formatter to the x-axis
ax.xaxis.set_major_formatter(FuncFormatter(kilounit_formatter))

# Hide every second tick label
for label in ax.get_xticklabels()[1::2]:
    label.set_visible(False)

# ax.set_title("Accuracy Box Plot and Average Accuracy vs. Prompt Length", fontsize=16)
ax.set_xlabel("Prompt Length", fontsize=14)
ax.set_ylabel("Accuracy", fontsize=14)
ax.legend(fontsize=12)
ax.grid(alpha=0.3)
plt.savefig(f"outputs/all_models_boxplot_length_{tpe}.pdf", format='pdf', bbox_inches='tight')
plt.close()

print("Box plot with average accuracy line saved!")