import argparse
import logging
import os
import json
import random
import socket
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

# python 12_efficiency_evaluate.py --prefill_len 8000 --decode_len 64 --num_iters 10 --batch_size 8 --test_mode whole --cuda_event --cuda_cache --results_file outputs/efficiency_1B_8batch_8k.txt


logging.basicConfig(
    format="%(levelname)s:%(asctime)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# List of your 5 models:
MODEL_NAMES = [
    "/lus/eagle/projects/LLM_multilang/DAM/DAM/LongEval_DAM_Exp/model/HanzhiZhang_H2O_LLaMA_1B",
    "/lus/eagle/projects/LLM_multilang/DAM/DAM/LongEval_DAM_Exp/model/HanzhiZhang_StreamingLLM_LLaMA_1B",
    "/lus/eagle/projects/LLM_multilang/DAM/DAM/LongEval_DAM_Exp/model/HanzhiZhang_MoA_LLaMA_1B",
    "/lus/eagle/projects/LLM_multilang/DAM/DAM/LongEval_DAM_Exp/model/HanzhiZhang_DAM_LLaMA_1B",
    "/lus/eagle/projects/LLM_multilang/DAM/DAM/LongEval_DAM_Exp/model/meta-llama_Llama-3.2-1B-Instruct"
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefill_len', type=int, required=True,
                        help='Number of tokens for the prompt/prefill stage.')
    parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='fp16',
                        help='Data type for model weights.')
    parser.add_argument('--num_iters', type=int, default=1,
                        help='Number of iterations to run.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size to test.')
    parser.add_argument('--decode_len', type=int, default=None,
                        help='Max decoding length. If None, only do prefilling.')
    parser.add_argument('--test_mode', type=str, choices=['decode', 'prefill', 'whole'], default='whole',
                        help="Benchmark stage: 'prefill' (prompt), 'decode' (incremental), or 'whole' (prefill+decode).")

    # Options to enable/disable some CUDA usage
    parser.add_argument('--cuda_event', action='store_true',
                        help="If set, measures wall-clock time with CUDA events.")
    parser.add_argument('--cuda_cache', action="store_true",
                        help="If set, calls torch.cuda.empty_cache and reset_peak_memory_stats.")

    parser.add_argument('--results_file', type=str, default='efficiency_results.txt',
                        help='Output text file to store all results.')

    # A simple JSONL file or any text with lines for reading prompts
    parser.add_argument('--test_file', type=str, default='longeval/evaluation/lines/testcases/8000_lines.jsonl',
                        help='Path to the file containing test prompts.')

    return parser.parse_args()

def fake_decode(model, prefill_output, max_len, batch_size):
    """
    Minimal incremental decoding loop. 
    Instead of sampling from logits, we do a random token to measure overhead.
    """
    result_tokens = []
    past_key_values = prefill_output.past_key_values
    device = next(model.parameters()).device

    # Start from the last token's argmax as a seed
    pred_token_idx = prefill_output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

    for _ in range(max_len - 1):
        with torch.inference_mode():
            outputs = model(input_ids=pred_token_idx, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values

        # For purely benchmarking, pick a random token from the logit dimension
        vocab_size = outputs.logits.shape[-1]
        random_token_id = random.randint(0, vocab_size - 1)
        pred_token_idx = torch.tensor([[random_token_id] for _ in range(batch_size)], 
                                      device=device, dtype=torch.long)

        result_tokens.append(random_token_id)

    return result_tokens

def run_benchmark(
    model_name: str,
    prefill_len: int,
    decode_len: int,
    batch_size: int,
    num_iters: int,
    test_mode: str,
    torch_dtype,
    test_file: str,
    use_cuda_event: bool = True,
    use_cuda_cache: bool = True
):
    """
    Runs the benchmark for a single model.
    Returns a dictionary of results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"========== Benchmarking Model: {model_name} ==========")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()

    # Prepare the prompt
    with open(test_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    test_cases = [json.loads(line) for line in lines]
    prompt = test_cases[0]["prompt"]
    prompt = "A chat between a user and an AI assistant. " + prompt
    prompt = prompt + " ASSISTANT:"
    prompt = prompt * 10  # repeat to enlarge if needed
    prompts = [prompt for _ in range(batch_size)]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        max_length=prefill_len,
        truncation=True
    ).to(device)
    input_ids = inputs["input_ids"]

    # Warm-up
    logger.info("Warming up (2 forward passes)...")
    with torch.inference_mode():
        for _ in range(2):
            _ = model(input_ids, use_cache=True)

    # Optional GPU management
    if use_cuda_cache:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # CUDA event timing
    if use_cuda_event:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    # Precompute prefill if decode only
    prefill_outputs_for_decode = []
    if test_mode == "decode":
        with torch.inference_mode():
            for _ in range(num_iters):
                out = model(input_ids, use_cache=True)
                prefill_outputs_for_decode.append(out)

    # Benchmark loop
    logger.info(f"Running test_mode={test_mode} for {num_iters} iterations.")
    with torch.inference_mode():
        if test_mode == "prefill":
            for i in range(num_iters):
                out = model(input_ids, use_cache=True)
                out = None
                logger.info(f"Iteration {i+1}/{num_iters} done (prefill).")

        elif test_mode == "decode":
            for i in range(num_iters):
                out = prefill_outputs_for_decode[i]
                _ = fake_decode(model, out, decode_len, batch_size)
                logger.info(f"Iteration {i+1}/{num_iters} done (decode).")

        else:  # whole
            for i in range(num_iters):
                out = model(input_ids, use_cache=True)
                if decode_len:
                    _ = fake_decode(model, out, decode_len, batch_size)
                logger.info(f"Iteration {i+1}/{num_iters} done (whole).")

    # Stop CUDA timing
    elapsed_ms = None
    if use_cuda_event:
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)  # in ms

    # Gather results
    results = {
        "model_name": model_name,
        "num_iters": num_iters,
        "batch_size": batch_size,
        "test_mode": test_mode,
        "prefill_len": prefill_len,
        "decode_len": decode_len if decode_len else 0,
        "elapsed_time_ms": None,
        "time_per_iter_ms": None,
        "time_per_token_ms": None,
        "throughput_tokens_s": None,
        "max_gpu_mem_mb": None
    }

    if elapsed_ms is not None:
        logger.info(f"Total elapsed time: {elapsed_ms:.2f} ms over {num_iters} iterations.")
        results["elapsed_time_ms"] = round(elapsed_ms, 2)
        avg_iter = elapsed_ms / num_iters
        logger.info(f"Average per iteration: {avg_iter:.2f} ms.")
        results["time_per_iter_ms"] = round(avg_iter, 2)

        # Additional metrics
        if test_mode == "prefill":
            tokens = prefill_len
            tpt = elapsed_ms / num_iters / batch_size / tokens
            logger.info(f"Time per token (prefill): {tpt:.2f} ms.")
            results["time_per_token_ms"] = round(tpt, 2)
        elif test_mode == "decode" and decode_len:
            tokens = decode_len
            tpt = elapsed_ms / num_iters / batch_size / tokens
            logger.info(f"Time per token (decode): {tpt:.2f} ms.")
            results["time_per_token_ms"] = round(tpt, 2)
            tps = batch_size * tokens * num_iters / (elapsed_ms / 1000.0)
            logger.info(f"Throughput: {tps:.2f} tokens/s.")
            results["throughput_tokens_s"] = round(tps, 2)
        elif test_mode == "whole" and decode_len:
            total_tokens = prefill_len + decode_len
            tpt = elapsed_ms / num_iters / batch_size / total_tokens
            logger.info(f"Time per token (whole): {tpt:.2f} ms.")
            results["time_per_token_ms"] = round(tpt, 2)
            tps = (batch_size * total_tokens * num_iters) / (elapsed_ms / 1000.0)
            logger.info(f"Throughput: {tps:.2f} tokens/s.")
            results["throughput_tokens_s"] = round(tps, 2)

    if use_cuda_cache:
        max_mem = torch.cuda.max_memory_allocated() // 2**20
        logger.info(f"Max GPU memory allocated: {max_mem} MB")
        results["max_gpu_mem_mb"] = int(max_mem)

    logger.info(f"========== Finished {model_name} ==========\n")
    return results

def main():
    args = parse_args()

    # Decide torch_dtype based on args
    if args.dtype == 'fp16':
        torch_dtype = torch.float16
    elif args.dtype == 'bf16':
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    all_results = []
    # Run evaluation on each of the 5 models
    for model_name in MODEL_NAMES:
        try:
            single_res = run_benchmark(
                model_name=model_name,
                prefill_len=args.prefill_len,
                decode_len=args.decode_len,
                batch_size=args.batch_size,
                num_iters=args.num_iters,
                test_mode=args.test_mode,
                torch_dtype=torch_dtype,
                test_file=args.test_file,
                use_cuda_event=args.cuda_event,
                use_cuda_cache=args.cuda_cache
            )
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"Out of Memory when processing model: {model_name}")
            single_res = {
                "model_name": model_name,
                "elapsed_time_ms": "OOM",
                "time_per_iter_ms": "OOM",
                "time_per_token_ms": "OOM",
                "throughput_tokens_s": "OOM",
                "max_gpu_mem_mb": "OOM"
            }
            # Clear CUDA cache to prevent memory leaks
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"An error occurred while processing model {model_name}: {e}")
            single_res = {
                "model_name": model_name,
                "elapsed_time_ms": "Error",
                "time_per_iter_ms": "Error",
                "time_per_token_ms": "Error",
                "throughput_tokens_s": "Error",
                "max_gpu_mem_mb": "Error"
            }
            torch.cuda.empty_cache()

        all_results.append(single_res)

        # Write/append partial results to file
        with open(args.results_file, "a", encoding="utf-8") as f_out:
            f_out.write(f"{json.dumps(single_res)}\n")

    # Finally, append a summary table
    table_header = (
        "\nSummary Table:\n"
        "---------------------------------------------------------------------------\n"
        " Model Name                          | Elapsed (ms) | ms/iter | ms/token | Thpt(tok/s) | MaxMem(MB)\n"
        "---------------------------------------------------------------------------\n"
    )

    table_lines = []
    for r in all_results:
        mname = r["model_name"]
        # Check if the result indicates an OOM or other error
        if r["elapsed_time_ms"] in ["OOM", "Error"]:
            table_lines.append(f" {mname:35s} | {'OOM':>11s} | {'OOM':>7s} | {'OOM':>8s} | {'OOM':>11s} | {'OOM':>10s}")
        else:
            elap = r["elapsed_time_ms"] or 0
            per_iter = r["time_per_iter_ms"] or 0
            per_token = r["time_per_token_ms"] or 0
            thpt = r["throughput_tokens_s"] or 0
            mem = r["max_gpu_mem_mb"] or 0
            table_lines.append(f" {mname:35s} | {elap:11.2f} | {per_iter:7.2f} | {per_token:8.2f} | {thpt:11.2f} | {mem:10d}")

    summary_table = table_header + "\n".join(table_lines) + "\n"

    with open(args.results_file, "a", encoding="utf-8") as f_out:
        f_out.write(summary_table)

    logger.info("All results appended to file. Done.")

if __name__ == "__main__":
    main()