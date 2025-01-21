import os
import re
import matplotlib.pyplot as plt

# List of model names
model_names = [
    "meta-llama_Llama-3.2-1B-Instruct", "HanzhiZhang_DAM_LLaMA_1B", "HanzhiZhang_H2O_LLaMA_1B", 
    "HanzhiZhang_MoA_LLaMA_1B", "HanzhiZhang_StreamingLLM_LLaMA_1B"
]
output_file = "outputs/all_models_lines.txt"

def extract_values(line):
    match = re.search(r"Accuracy: ([0-9\.]+), Ave Prompt Length: ([0-9\.]+)", line)
    if match:
        accuracy = float(match.group(1))
        length = float(match.group(2)) / 1000
        return accuracy, length
    return None, None

# Process each model
all_results = {}
for model_name in model_names:
    folder_path = f"LongEval/evaluation/lines_1/predictions/{model_name}/"
    model_results = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            line_number = int(re.search(r"(\d+)_response", filename).group(1))
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    accuracy, length = extract_values(last_line)
                    if accuracy is not None and length is not None:
                        model_results.append((line_number, f"{int(length)}", accuracy))
    model_results.sort(key=lambda x: x[0])
    all_results[model_name] = model_results

# Write results to output file
os.makedirs("outputs", exist_ok=True)
with open(output_file, "w") as f:
    f.write("line_number length LLaMA DAM H2O MoA StreamingLLM\n")
    for line_number in sorted(set(num for model in all_results.values() for num, _, _ in model)):
        line_data = [f"{line_number}"]
        length_written = False
        for model_name in model_names:
            model_data = {num: (length, accuracy) for num, length, accuracy in all_results[model_name]}
            if line_number in model_data:
                length, accuracy = model_data[line_number]
                if not length_written:
                    line_data.append(f"{length}k")
                    length_written = True
                line_data.append(f"{accuracy}")
            else:
                line_data.append("-")
        f.write(" ".join(map(str, line_data)) + "\n")

print(f"Results saved to {output_file}")

# Load data from file
data_file = output_file
with open(data_file, 'r') as file:
    lines_meta, llama_meta, dam_meta, h2o_meta, moa_meta, streamingllm_meta = [], [], [], [], [], []
    next(file)  # Skip header
    for line in file:
        parts = line.strip().split()
        lines_meta.append(int(parts[0]))
        llama_meta.append(float(parts[2]))
        dam_meta.append(float(parts[3]))
        h2o_meta.append(float(parts[4]))
        moa_meta.append(float(parts[5]))
        streamingllm_meta.append(float(parts[6]))

# Filtering data to only include lines ≤ 3100
filtered_indices_meta = [i for i, line in enumerate(lines_meta) if line <= 3100]
lines_filtered_meta = [lines_meta[i] for i in filtered_indices_meta]
llama_filtered_meta = [llama_meta[i] for i in filtered_indices_meta]
dam_filtered_meta = [dam_meta[i] for i in filtered_indices_meta]
h2o_filtered_meta = [h2o_meta[i] for i in filtered_indices_meta]
moa_filtered_meta = [moa_meta[i] for i in filtered_indices_meta]
streamingllm_filtered_meta = [streamingllm_meta[i] for i in filtered_indices_meta]

# Plotting and saving figure
plt.figure(figsize=(12, 8))
plt.plot(lines_filtered_meta, llama_filtered_meta, label="LLaMA", color='black', linewidth=2, linestyle='solid')
plt.plot(lines_filtered_meta, dam_filtered_meta, label="DAM", color='red', linewidth=1.5, linestyle='dotted')
plt.plot(lines_filtered_meta, h2o_filtered_meta, label="H2O", color='purple', linewidth=1.5, linestyle='dotted')
plt.plot(lines_filtered_meta, moa_filtered_meta, label="MoA", color='blue', linewidth=1.5, linestyle='dotted')
plt.plot(lines_filtered_meta, streamingllm_filtered_meta, label="StreamingLLM", color='green', linewidth=1.5, linestyle='dotted')

plt.title("Accuracy vs. Lines/Length", fontsize=16)
plt.xlabel("Lines", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.savefig("outputs/all_models_lines.pdf", format='pdf')
plt.close()

print("Plot saved to outputs/all_models_lines.pdf")
