# easy-llm
llm code demo for begginer.

## Introduction

Welcome to the easy-llm Repository, a beginner-friendly platform that provides a comprehensive introduction to fine-tuning and inference of the LLM (Language Model) using practical code examples. Whether you're new to machine learning, natural language processing, or you're simply looking to expand your existing skills, this repository is an excellent starting point.

## Repository Content

This repository contains a series of  Python scripts that demonstrate how to fine-tune and perform inference on an LLM model. It includes:

- Fine-tuning a pre-trained LLM model on a custom dataset.
- Using a fine-tuned LLM model to generate predictions (inference).
- Detailed comments explaining each step of the process.

## Milestone


| Status | Base Model | Finetune Method | Dataset | Desc | Code  
| :----: | :--------: | :--------------: | :---------: | :---------: |  :--: 
| <input type="checkbox" checked> | tiiuae/falcon-7b |  QLoRA  |  heliosbrahma/mental_health_chatbot_dataset  | finetune demo using lora with SFT |  [Code](./falcon-7b/finetune_v1.py) 
| <input type="checkbox" checked> | tiiuae/falcon-7b |  QLoRA  |  -   | inference demo using lora  |  [Code](./falcon-7b/inference_v1.py) 
|  <input type="checkbox" checked>  | tiiuae/falcon-7b |  QLoRA  |  b-mc2/sql-create-context   | finetune demo using lora |  [Code](./falcon-7b/finetune_v2.py) 
|  <input type="checkbox" checked>  | tiiuae/falcon-7b |  QLoRA  |  -        |       |  [Code](./falcon-7b/inference_v2.py) 
<!-- |  <input type="checkbox" unchecked>  | tiiuae/falcon-7b |  QLoRA  |  custom   | finetune demo using lora with custom json format dataset on multi server GPUs|  
|  <input type="checkbox" unchecked>  | tiiuae/falcon-7b |  QLoRA  |  -        |       |   -->


