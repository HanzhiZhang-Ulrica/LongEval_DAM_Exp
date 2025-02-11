"""
Enhanced Benchmarking Script for DAM and Original LLaMA Models

Usage:
python 12_efficiency_evaluate.py \
            --prefill_len 65536 \
            --decode_len 65536 \
            --num_iters 10 \
            --batch_size 32 \
            --model_names "model/HanzhiZhang_DAM_LLaMA_1B,model/meta-llama_Llama-3.2-1B-Instruct,model/HanzhiZhang_H2O_LLaMA_1B,model/HanzhiZhang_MoA_LLaMA_1B, model/HanzhiZhang_StreamingLLM_LLaMA_1B" \
            --test_file "longeval/evaluation/lines/testcases/200_lines.jsonl"
"""
import argparse
import logging
import os
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tabulate import tabulate  # For formatted table output

# Configure Logging
def setup_logging(output_file_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set to INFO to reduce verbosity

    formatter = logging.Formatter(
        fmt="%(levelname)s:%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(output_file_path, mode='w')  # Overwrite the file each run
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

logger = None  # Initialize logger to be set up later

def parse_args():
    parser = argparse.ArgumentParser(description="Simple Benchmarking for DAM and LLaMA Models with GPU Memory Reporting.")
    parser.add_argument('--prefill_len', type=int, required=True, help='Number of tokens for the prompt/prefill stage.')
    parser.add_argument('--decode_len', type=int, required=True, help='Number of tokens to decode/generate.')
    parser.add_argument('--num_iters', type=int, default=1, help='Number of iterations to run.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size to test.')
    parser.add_argument('--model_names', type=str, required=True, help='Comma-separated list of model paths or names.')
    parser.add_argument('--test_file', type=str, required=True, help='Path to the file containing test prompts.')
    return parser.parse_args()

def load_models(model_name, torch_dtype, device):
    models = []
    tokenizers = []
    logger.info(f"Loading model and tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        ).to(device)
        model.eval()
        models.append(model)
        tokenizers.append(tokenizer)
        logger.info(f"Successfully loaded {model_name}")
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
    return models, tokenizers

def load_prompts(test_file, batch_size):
    try:
        with open(test_file, 'r', encoding="utf-8") as f:
            lines = f.readlines()
        test_cases = [json.loads(line) for line in lines if line.strip()]
        if not test_cases:
            raise ValueError("No valid test cases found.")
        prompt = test_cases[0].get("prompt", "")
        if not prompt:
            raise ValueError("Prompt is empty in the first test case.")
        prompt = "A chat between a user and an AI assistant. " + prompt + " ASSISTANT:"
        prompts = [prompt for _ in range(batch_size)]
        return prompts
    except Exception as e:
        logger.error(f"Error loading prompts from {test_file}: {e}")
        return None

def benchmark_model(model, tokenizer, prompts, prefill_len, decode_len, num_iters, device):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, max_length=prefill_len, truncation=True).to(device)
    input_ids = inputs["input_ids"]

    generation_kwargs = {
        "max_new_tokens": decode_len,
        "do_sample": False,  # Disable sampling for consistent benchmarking
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id  # Ensure pad_token_id is set
    }

    logger.info("Warming up (2 forward passes)...")
    with torch.no_grad():
        for _ in range(2):
            _ = model.generate(input_ids, **generation_kwargs)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    logger.info(f"Running benchmark for {num_iters} iterations...")
    total_time = 0.0
    with torch.no_grad():
        for i in range(num_iters):
            start_time = time.time()
            _ = model.generate(input_ids, **generation_kwargs)
            end_time = time.time()
            iter_time = (end_time - start_time) * 1000  # Convert to ms
            logger.info(f"Iteration {i+1}/{num_iters}: {iter_time:.2f} ms")
            total_time += iter_time

    max_mem_mb = None
    if device.type == 'cuda':
        max_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    avg_time = total_time / num_iters
    tokens_processed = decode_len * num_iters * len(prompts)
    throughput = tokens_processed / (total_time / 1000.0)

    return avg_time, throughput, max_mem_mb

def main():
    global logger

    args = parse_args()
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "model_efficiency.txt")

    logger = setup_logging(output_file_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    logger.info(f"Using device: {device}")

    model_names = [name.strip() for name in args.model_names.split(',')]
    results = []

    for model_name in model_names:
        models, tokenizers = load_models(model_name, torch_dtype, device)
        if not models:
            logger.error(f"Failed to load model: {model_name}")
            continue

        prompts = load_prompts(args.test_file, args.batch_size)
        if not prompts:
            logger.error("Failed to load prompts. Exiting.")
            return

        for model, tokenizer in zip(models, tokenizers):
            logger.info(f"========== Benchmarking Model: {model_name} ==========")
            avg_time, throughput, max_mem_mb = benchmark_model(
                model, tokenizer, prompts, args.prefill_len, args.decode_len, args.num_iters, device
            )
            logger.info(f"Average Time per Iteration: {avg_time:.2f} ms")
            logger.info(f"Throughput: {throughput:.2f} tokens/s")
            if max_mem_mb is not None:
                logger.info(f"Max GPU Memory Allocated: {max_mem_mb:.2f} MB")
            logger.info(f"========== Finished Benchmarking {model_name} ==========\n")

            results.append([model_name, avg_time, throughput, max_mem_mb])

            del model
            del tokenizer
            torch.cuda.empty_cache()

    # Summarize results in a table
    summary_table = tabulate(results, headers=["Model", "Avg Time (ms)", "Throughput (tokens/s)", "Max GPU Mem (MB)"], tablefmt="grid")
    logger.info("\n========== Final Benchmark Summary ==========\n")
    logger.info(f"\n{summary_table}\n")

    with open(output_file_path, "a") as f:
        f.write("\n========== Final Benchmark Summary ==========\n")
        f.write(summary_table + "\n")

if __name__ == "__main__":
    main()