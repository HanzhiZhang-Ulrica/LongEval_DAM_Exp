import os
import re

# List of model names
model_names = ["meta-llama_Llama-3.2-1B-Instruct", "HanzhiZhang_DAM_LLaMA_1B", "HanzhiZhang_H2O_LLaMA_1B", "HanzhiZhang_MoA_LLaMA_1B", "HanzhiZhang_StreamingLLM_LLaMA_1B"]
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
with open(output_file, "w") as f:
    f.write("line_number length meta-llama_Llama-3.2-1B-Instruct HanzhiZhang_DAM_LLaMA_1B HanzhiZhang_H2O_LLaMA_1B HanzhiZhang_MoA_LLaMA_1B HanzhiZhang_StreamingLLM_LLaMA_1B\n")
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

