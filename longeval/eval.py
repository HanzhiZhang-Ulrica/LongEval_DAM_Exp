import argparse
import os
import json
from tqdm import tqdm

import torch
import numpy as np

from fastchat.model import get_conversation_template
from utils import maybe_monkey_patch, get_output_dir, longeval_load_model, load_testcases, test_topics_one_sample, test_lines_one_sample 


def longeval_test(model, tokenizer, output_dir, args):
    if args.task == "topics":
        for num_topics in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
            print(f"************ Start testing {num_topics} topics per prompt ***********")
            avg_length = 0

            test_file = os.path.join(args.test_dir, f"topics/testcases/{num_topics}_topics.jsonl")
            output_file = os.path.join(output_dir, f"{num_topics}_response.txt")
            
            test_cases = load_testcases(test_file)
            for idx, test_case in tqdm(enumerate(test_cases)):
                _, prompt_length, summary = test_topics_one_sample(model=model, tokenizer=tokenizer, test_case=test_case, output_file=output_file, idx=idx, args=args)
                avg_length += prompt_length / len(test_cases)

            print(f"************ Finish testing {num_topics} topics per prompt with average prompt length {avg_length} ************")
            if args.eval_shortest_only:
                break
            
    elif args.task == "lines":
        for num_lines in [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000, 6100, 6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000, 7100, 7200, 7300, 7400, 7500, 7600, 7700, 7800, 7900, 8000, 8100]:
            print(f"************ Start testing {num_lines} lines per LRT prompt ************")
            test_file = os.path.join(args.test_dir, f"lines/testcases/{num_lines}_lines.jsonl")
            
            output_file = os.path.join(output_dir, f"{num_lines}_response.txt")
            num_correct = 0
            avg_length = 0

            test_cases = load_testcases(test_file)
            for idx, test_case in tqdm(enumerate(test_cases)):
                correct, prompt_length, summary = test_lines_one_sample(model=model, tokenizer=tokenizer, test_case=test_case, output_file=output_file, idx=idx, args=args)
                avg_length += prompt_length / len(test_cases)
                num_correct += correct
            accuracy = num_correct / len(test_cases)

            with open(output_file, "a+") as f:
                f.write(f"Accuracy: {accuracy}, Ave Prompt Length: {avg_length}")

            print(f"************ Finish testing {num_lines} lines per prompt with average prompt length {avg_length}, accuracy: {accuracy} ************")
            if args.eval_shortest_only:
                break
    else:
        print(f"Unsupported task: {args.task}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True, help="model path")
    parser.add_argument("--task", type=str, required=True, help="Which evaluation task to use. currently support [topics, lines]")
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--max_gpu_memory", type=int, default=40, help="max per gpu memory in GiB. A100 is 40 or 80.")
    parser.add_argument("--longchat_flash_attn", action='store_true', help="Only apply to longchat models. Whether to enable flash attention to save memory, but slower.")
    parser.add_argument("--longchat_ratio", type=int, default=8, help="Only apply to longchat models. Use ratio=8 for 16K context length model. Only ratio=8 is supported now.")
    parser.add_argument("--eval_shortest_only", action='store_true', default=0, help="Only eval the shortest case for illustration purpose")
    parser.add_argument("--test_dir", type=str, default="evaluation", help="Directory of the testcases")
    parser.add_argument("--framework", type=str, default=None, help="Framework for serving")
    parser.add_argument("--count", type=str, default="1", help="Iteration Count")
    args = parser.parse_args()

    maybe_monkey_patch(args)
    output_dir = get_output_dir(args)

    model, tokenizer = longeval_load_model(args)
    longeval_test(model, tokenizer, output_dir, args)
