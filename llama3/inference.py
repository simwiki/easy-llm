import copy
import json
import os
import sys
import time
import fire
import torch

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from peft import PeftModel
from transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaConfig
from torch.utils.data import Dataset


PROMPT_DICT = {
    "prompt_with_context": (
        "You are a powerful text-to-SQL model. Your job is to answer questions correctly about a database. You are given a question, paired with a context regarding one or more tables that provides further info about database schema. "
        "You must output the SQL query that correctly answers the question with specific field in context if provided."
        "### Question:\n{question}\n\n### Context:\n{context}\n\n### Answer:"
    ),
    "prompt_without_context": (
        "You are a powerful text-to-SQL model. Your job is to answer questions correctly about a database. You are given a question. "
        "You must output the SQL query that correctly answers the question with specific field in context if provided."
        "### Question:\n{question}\n\n### Answer:"
    ),
}



red = "31"
green = "32"
blue = "34"


def print_colored(text, color_code="red"):
    return print(f"\033[1;{color_code};40m{text}\033[0m")


# Function to load the main model for text generation
def load_model(model_name, quantization, use_fast_kernels):
    print(f"use_fast_kernels={use_fast_kernels}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa" if use_fast_kernels else None,
    )
    return model


# Function to load the PeftModel for performance optimization
def load_peft_model(model, peft_model):
    peft_model = PeftModel.from_pretrained(model, peft_model)
    return peft_model


# Loading the model from config to load FSDP checkpoints into that
def load_llama_from_config(config_path):
    model_config = LlamaConfig.from_pretrained(config_path) 
    model = LlamaForCausalLM(config=model_config)
    return model


def main(
    model_name,
    peft_model: str=None,
    quantization: bool=False,
    dataset_path: str=None,
    max_new_tokens =100, #The maximum numbers of tokens to generate
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs
):
    model = load_model(model_name, quantization, use_fast_kernels)
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def inference(user_prompt, temperature, top_p, top_k, max_new_tokens, **kwargs,):
        # Set the seeds for reproducibility
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)        
        batch = tokenizer(user_prompt, padding='max_length', truncation=True, max_length=max_padding_length, return_tensors="pt")
        batch = {k: v.to("cuda") for k, v in batch.items()}

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs
            )
        e2e_inference_time = (time.perf_counter()-start)*1000
        print(f"the inference time is {e2e_inference_time} ms")
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_text

    # dataset = json.load(open(dataset_path))
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle().select(range(10))
    batch_prompt = []
    for data in dataset:
        user_prompt = PROMPT_DICT["prompt_with_context"].format_map(data)
        batch_prompt.append(user_prompt)
        output_tune = inference(user_prompt, temperature, top_p, top_k, max_new_tokens)
        output_tune = output_tune[len(user_prompt):]
        print('\n')
        print('*'*120)
        print_colored(f"User prompt:\n{user_prompt}", red)
        print('-'*100)
        print_colored(f'[answer]:>\n{data["answer"]}', green)
        print('-'*100)
        print_colored(f'[tune model]:>\n{output_tune}', blue)
        print('-'*100)
        print('*'*120)
        print('\n\n')


if __name__ == "__main__":
    """
    usage: 
    python inference.py --model_name /service/llm/Meta-Llama-3-8B --quantization \
        --peft_model ./output/peft/llama3-8B-lora-v1 \
        --dataset_path ./fake_sql.json
    """
    fire.Fire(main)

