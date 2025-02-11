#!/usr/bin/env python
"""
Simplified Benchmarking Script for Model Efficiency

This script benchmarks the efficiency of Hugging Face models by measuring GPU memory usage,
throughput (tokens per second), and latency (time taken to generate tokens).

Usage:
    python 11_model_efficiency.py \
        --model-path model \
        --max-new-tokens 128 \
        --batch-size 1 \
        --device cuda
"""

import os
import time
import torch
import argparse
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure Logging
logging.basicConfig(
    format="%(levelname)s:%(asctime)s %(message)s",
    level=logging.INFO,  # Set to INFO for general logs
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def create_prompt(prompt_length, batch_size, tokenizer, device):
    """
    Creates a dummy prompt of specified length filled with the pad token.
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids = torch.full(
        (batch_size, prompt_length),
        fill_value=tokenizer.pad_token_id,
        dtype=torch.long,
        device=device
    )
    return {"input_ids": input_ids}

def benchmark_model(
    model_path,
    prompt_length,
    batch_size=1,
    max_new_tokens=128,
    device="cuda",
    trust_remote_code=False
):
    """
    Benchmarks a single model by generating tokens and measuring performance metrics.
    """
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        model.eval()

        logger.info(f"\nLoaded model from: {model_path}")
        logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

        # Create prompt
        inputs = create_prompt(prompt_length, batch_size, tokenizer, device)

        # Clear CUDA cache and reset memory stats
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        # Measure GPU memory before inference
        if device == "cuda":
            start_mem = torch.cuda.memory_allocated(device)

        # Synchronize and start timer
        if device == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        # Generate tokens
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=False  # Disable scores to simplify
            )

        # Synchronize and end timer
        if device == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()

        # Measure GPU memory after inference
        if device == "cuda":
            end_mem = torch.cuda.memory_allocated(device)
            peak_mem = torch.cuda.max_memory_allocated(device)
        else:
            peak_mem = None  # GPU memory not applicable for CPU

        # Calculate metrics
        elapsed_time = end_time - start_time  # in seconds
        total_generated_tokens = max_new_tokens * batch_size
        throughput = total_generated_tokens / elapsed_time if elapsed_time > 0 else float('inf')
        scalability_score = throughput / (peak_mem / 1e6) if peak_mem and peak_mem != 0 else float('inf')

        metrics = {
            "Model": os.path.basename(model_path),
            "Sequence Length": prompt_length,
            "GPU Memory (MB)": peak_mem / 1e6 if peak_mem else "N/A",
            "Throughput (tok/s)": round(throughput, 2),
            "Latency (s)": round(elapsed_time, 4),
            "Scalability Score": round(scalability_score, 4) if peak_mem else "N/A"
        }

        # Log metrics
        logger.info(f"Sequence Length: {prompt_length}")
        logger.info(f"GPU Memory Usage: {metrics['GPU Memory (MB)']} MB")
        logger.info(f"Throughput: {metrics['Throughput (tok/s)']} tokens/sec")
        logger.info(f"Latency: {metrics['Latency (s)']} seconds")
        logger.info(f"Scalability Score: {metrics['Scalability Score']}")

        return metrics

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        logger.error("CUDA OOM - Skipping this configuration")
        return {
            "Model": os.path.basename(model_path),
            "Sequence Length": prompt_length,
            "GPU Memory (MB)": "OOM",
            "Throughput (tok/s)": "OOM",
            "Latency (s)": "OOM",
            "Scalability Score": "OOM"
        }
    except Exception as e:
        logger.error(f"Error benchmarking model {model_path} with sequence length {prompt_length}: {e}")
        return {
            "Model": os.path.basename(model_path),
            "Sequence Length": prompt_length,
            "GPU Memory (MB)": "ERROR",
            "Throughput (tok/s)": "ERROR",
            "Latency (s)": "ERROR",
            "Scalability Score": "ERROR"
        }

def main():
    parser = argparse.ArgumentParser(description="Benchmark Hugging Face models for efficiency.")
    parser.add_argument("--model-path", type=str, required=True, help="Root directory containing model subdirectories.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Tokens to generate beyond the prompt.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for generation.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run the model on.")
    args = parser.parse_args()

    # List of your models
    model_dirs = [
        "HanzhiZhang_H2O_LLaMA_1B",
        "HanzhiZhang_StreamingLLM_LLaMA_1B",
        "HanzhiZhang_MoA_LLaMA_1B",
        "HanzhiZhang_DAM_LLaMA_1B",
        "meta-llama_Llama-3.2-1B-Instruct",
    ]

    # Define sequence lengths within model's maximum context window
    # Common LLaMA models have a max context length of 2048 or 4096
    sequence_lengths = [512, 1024, 2048, 4096]

    # Iterate over each model and sequence length
    for model_dir in model_dirs:
        trust_remote_code = ("DAM" in model_dir)  # Adjust based on your model's requirements
        model_full_path = os.path.join(args.model_path, model_dir)

        logger.info(f"\nStarting benchmarks for model: {model_dir}")

        for seq_len in sequence_lengths:
            logger.info(f"\nBenchmarking sequence length: {seq_len}")
            metrics = benchmark_model(
                model_path=model_full_path,
                prompt_length=seq_len,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                trust_remote_code=trust_remote_code
            )

            # Print metrics in a structured format
            print("\nBenchmark Results:")
            print("===================")
            for key, value in metrics.items():
                print(f"{key}: {value}")
            print("===================\n")

    logger.info("\nAll benchmarks completed.")

if __name__ == "__main__":
    main()
