import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

hf_token = "<token>"

def download_models(model_names, save_dir):
    """
    Downloads the specified models to a given directory.

    Args:
        model_names (list of str): List of Hugging Face model names to download.
        save_dir (str): Directory to save the models.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for model_name in model_names:
        print(f"Downloading model: {model_name}")
        model_path = os.path.join(save_dir, model_name.replace('/', '_'))

        # Download and save model
        model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
        model.save_pretrained(model_path)

        # Download and save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        tokenizer.save_pretrained(model_path)

        print(f"Model saved to: {model_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Hugging Face models.")
    parser.add_argument("--model-path", type=str, default="model", required=True, help="Directory to save the models.")
    args = parser.parse_args()

    models = [
        "HanzhiZhang/H2O_LLaMA_3B",
        "HanzhiZhang/StreamingLLM_LLaMA_3B",
        "HanzhiZhang/MoA_LLaMA_3B",
        "HanzhiZhang/DAM_LLaMA_3B",
        "meta-llama/Llama-3.2-3B-Instruct",
    ]

    download_models(models, args.model_path)
