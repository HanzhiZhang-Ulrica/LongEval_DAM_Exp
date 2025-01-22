import os
import time
import torch
import argparse
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM

def create_prompt(prompt_length, batch_size, tokenizer, device):
    """
    Create a prompt of exactly `prompt_length` tokens. This avoids the confusion of
    using a repeated string like 'hello ' which might tokenize into more or fewer
    tokens than expected.
    """
    if tokenizer.pad_token_id is None:
        # Some LLaMA-based models might not have an official pad_token_id set:
        tokenizer.pad_token = tokenizer.eos_token

    # Here we just fill with the pad_token_id, but it could be anything you like.
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
    Benchmark a Hugging Face model for GPU memory usage, throughput, latency, and scalability.

    Args:
        model_path (str): Local path to the saved model.
        prompt_length (int): Exact length of the *input prompt* in tokens.
        batch_size (int): Number of inputs (prompts) to process at once.
        max_new_tokens (int): How many tokens to generate beyond the prompt.
        device (str): Device to run the model on (default: 'cuda').
        trust_remote_code (bool): Whether to allow custom code from model repo.

    Returns:
        dict: Performance metrics including memory, throughput, latency, and scalability.
    """
    # ---------- Load tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------- Load model ----------
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model.to(device)
    model.eval()

    # Print some debug info to confirm the correct model is loaded
    print(f"\nLoaded model from: {model_path}")
    # print(f"Model config: {model.config}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # ---------- Create the prompt (exact token length) ----------
    inputs = create_prompt(prompt_length, batch_size, tokenizer, device)

    # ---------- Clear CUDA memory stats ----------
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # ---------- Run generation ----------
    start_mem = torch.cuda.memory_allocated(device)
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

    end_time = time.time()
    end_mem = torch.cuda.memory_allocated(device)
    peak_mem = torch.cuda.max_memory_allocated(device)

    # ---------- Compute metrics ----------
    elapsed_time = end_time - start_time

    # Total tokens processed = (prompt_length + generated_tokens) * batch_size
    # But we are just using the final sequence length to represent throughput:
    # (You can change the calculation if you want to separate prompt vs new tokens.)
    total_tokens = outputs.shape[1] * batch_size

    throughput = total_tokens / elapsed_time if elapsed_time > 0 else float('inf')
    scalability_score = total_tokens / peak_mem if peak_mem != 0 else float('inf')

    return {
        "gpu_memory_MB": peak_mem / 1e6,
        "throughput_tokens_per_sec": throughput,
        "latency_sec": elapsed_time,
        "scalability_score": scalability_score
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Hugging Face models.")
    parser.add_argument("--model-path", type=str, required=True, 
                        help="Root directory containing model subdirectories.")
    parser.add_argument("--max-new-tokens", type=int, default=512, 
                        help="Tokens to generate beyond the prompt.")
    parser.add_argument("--batch-size", type=int, default=1, 
                        help="Batch size for generation.")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to run the model on (cpu or cuda).")
    args = parser.parse_args()

    model_dirs = [
        "HanzhiZhang_H2O_LLaMA_1B",
        "HanzhiZhang_StreamingLLM_LLaMA_1B",
        "HanzhiZhang_MoA_LLaMA_1B",
        "HanzhiZhang_DAM_LLaMA_1B",
        "meta-llama_Llama-3.2-1B-Instruct",
    ]

    # NOTE: This is now truly "tokens" in the input_ids, not repeated strings.
    sequence_lengths = [2000, 4000, 8000, 16000, 32000, 64000, 128000]

    # Store results in a dict for reference
    results = {}

    # Ensure 'outputs' folder exists
    os.makedirs("outputs", exist_ok=True)
    output_file = "outputs/model_efficiency.txt"

    with open(output_file, "w") as f:
        # Write CSV header
        f.write("Model,Seq Len,GPU Memory (MB),Throughput (tok/s),Latency (s)\n")

        for model_dir in model_dirs:
            # Decide whether to trust remote code based on 'DAM' in the name
            trust_remote_code = ("DAM" in model_dir)

            model_full_path = os.path.join(args.model_path, model_dir)
            results[model_dir] = {}

            for seq_len in sequence_lengths:
                try:
                    metrics = benchmark_model(
                        model_path=model_full_path,
                        prompt_length=seq_len,
                        batch_size=args.batch_size,
                        max_new_tokens=args.max_new_tokens,
                        device=args.device,
                        trust_remote_code=trust_remote_code
                    )
                    results[model_dir][seq_len] = metrics
                    f.write(
                        f"{model_dir},{seq_len},"
                        f"{metrics['gpu_memory_MB']:.2f},"
                        f"{metrics['throughput_tokens_per_sec']:.2f},"
                        f"{metrics['latency_sec']:.2f}\n"
                    )

                except Exception as e:
                    # Print the full traceback to stdout
                    print(f"\nERROR benchmarking {model_dir} with seq_len={seq_len}:")
                    traceback.print_exc()

                    results[model_dir][seq_len] = {"error": str(e)}
                    f.write(f"{model_dir},{seq_len},ERROR,ERROR,ERROR\n")

                # After each test, delete the model reference and empty cache 
                # to reduce carry-over from one sequence length to the next.
                # (If the function returns, 'model' is out of scope already, 
                #  but let's be extra sure.)
                torch.cuda.empty_cache()

    print("\nBenchmark results saved to outputs/model_efficiency.txt")
