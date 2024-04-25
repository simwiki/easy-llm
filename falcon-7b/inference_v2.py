import os
import torch
import wandb
from datetime import datetime
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq, Trainer, TrainingArguments, GenerationConfig
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, PeftConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer
import warnings

warnings.filterwarnings("ignore")

device='cuda' if torch.cuda.is_available() else 'cpu'


red = "31"
green = "32"
blue = "34"


def colored(text, color_code):
    return f"\033[1;{color_code};40m{text}\033[0m"


def generate_prompt_sql(input_question, context, output=""):
    """
    Generates a prompt for fine-tuning the LLM model for text-to-SQL tasks.

    Parameters:
        input_question (str): The input text or question to be converted to SQL.
        context (str): The schema or context in which the SQL query operates.
        output (str, optional): The expected SQL query as the output.

    Returns:
        str: A formatted string serving as the prompt for the fine-tuning task.
    """
    return f"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables. 

You must output the SQL query that answers the question.

### Input:
{input_question}

### Context:
{context}

### Response:
{output}"""


def tokenize_batch(data_points, add_eos_token=True, train_on_inputs=False, cutoff_len=512) -> dict:
    """
    Tokenizes a batch of SQL related data points consisting of questions, context, and answers.

    Parameters:
        data_points (dict): A batch from the dataset containing 'question', 'context', and 'answer'.
        add_eos_token (bool): Whether to add an EOS token at the end of each tokenized sequence.
        cutoff_len (int): The maximum length for each tokenized sequence.

    Returns:
        dict: A dictionary containing tokenized 'input_ids', 'attention_mask', and 'labels'.
    """
    try:
        question = data_points["question"]
        context = data_points["context"]
        answer = data_points["answer"]
        if train_on_inputs:
            user_prompt = generate_prompt_sql(question, context)
            tokenized_user_prompt = tokenizer(
                user_prompt,
                truncation=True,
                max_length=cutoff_len,
                padding=False,
                return_tensors=None,
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            if add_eos_token:
                user_prompt_len -= 1

        combined_text = generate_prompt_sql(question, context, answer)
        tokenized = tokenizer(
            combined_text,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (tokenized["input_ids"][-1] != tokenizer.eos_token_id and add_eos_token and len(tokenized["input_ids"]) < cutoff_len):
            tokenized["input_ids"].append(tokenizer.eos_token_id)
            tokenized["attention_mask"].append(1)
        tokenized["labels"] = tokenized["input_ids"].copy()

        if train_on_inputs:
            tokenized["labels"] = [-100] * user_prompt_len + tokenized["labels"][user_prompt_len:]

        return tokenized
    except Exception as e:
        print.error(f"Error in batch tokenization: {e}, Line: {e.__traceback__.tb_lineno}")
        raise e


def prepare_data(data, val_set_size=100):
    train_dataset_dir = "./data/sql_context_train"
    valid_dataset_dir = "./data/sql_context_valid"
    if not os.path.exists(train_dataset_dir):
        os.makedirs(train_dataset_dir)
    if not os.path.exists(valid_dataset_dir):
        os.makedirs(valid_dataset_dir)
    try:
        if not os.path.exists(os.path.join(train_dataset_dir, "dataset_info.json")) or not os.path.exists(os.path.join(valid_dataset_dir, "dataset_info.json")):
            train_val_split = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
            train_data = train_val_split["train"].shuffle().map(tokenize_batch)
            valid_data = train_val_split["test"].shuffle().map(tokenize_batch)
            train_data.save_to_disk(train_dataset_dir)
            valid_data.save_to_disk(valid_dataset_dir)
        else:
            train_data = load_from_disk(train_dataset_dir)
            valid_data = load_from_disk(valid_dataset_dir)

        train_data = train_data.select(range(20))
        valid_data = valid_data.select(range(10))
        print(data)
        print(train_data)
        print(valid_data)
        return train_data, valid_data
    except Exception as e:
        print(f"Error in preparing data: {e}, Line: {e.__traceback__.tb_lineno}")
        raise e


model_name = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True)
print(f"Using tokenizer: {tokenizer.__class__.__name__}")
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

DATA_PATH = "b-mc2/sql-create-context"
data = load_dataset(DATA_PATH, num_proc=32)
train_data, val_data = prepare_data(data)


def load_model(use_lora=True):
    # load_8bit = False
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     load_in_8bit=load_8bit,
    #     torch_dtype=torch.float16,
    #     # quantization_config=bnb_config,         
    #     device_map="auto",                      
    # )
    # if use_lora:
    #     PEFT_MODEL_NAME = "./falcon_7b_lora_v2/checkpoint-200"
    #     model = PeftModel.from_pretrained(
    #         model,
    #         PEFT_MODEL_NAME,
    #         torch_dtype=torch.float16,
    #     )
    # # if not load_8bit:
    # #     model.half()  # seems to fix bugs for some users.
    # model.eval()
    PEFT_MODEL_NAME = "./falcon_7b_lora_v2/checkpoint-200"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                       # 表示以4位精度加载模型
        bnb_4bit_quant_type="nf4",               # 预训练模型应该以4位NF格式进行量化
        bnb_4bit_use_double_quant=True,          # 采用QLoRA论文中提到的双量化
        bnb_4bit_compute_dtype=torch.float16,    # 在计算过程中，预训练模型应以BF16格式加载
    )
    config = PeftConfig.from_pretrained(PEFT_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        return_dict=True,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    if use_lora:
        model = PeftModel.from_pretrained(model, PEFT_MODEL_NAME, is_trainable=False)
    model.eval()

    return model


def generate_output(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    # tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    # print(tokens)
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.3,
        top_p=0.85,
        top_k=40,
        num_beams=1,
        max_new_tokens=600,
        repetition_penalty=1.2,
        pad_token_id = 0,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=256,
        )

    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    response_text = output.split("### Response:")[1].strip()
    return response_text


model_base = load_model(use_lora=False)
model_tune = load_model(use_lora=True)


for row_dict in val_data:
    prompt = generate_prompt_sql(row_dict["question"], row_dict["context"])
    output_base = generate_output(model=model_base, prompt=prompt)
    output_tune = generate_output(model=model_tune, prompt=prompt)
    print('\n')
    print('*'*100)
    print(colored(f'[question]:>\n{row_dict["question"]}', blue))
    print(colored(f'[answer]:>\n{row_dict["answer"]}', blue))
    print('-'*100)
    print(colored(f'[base model]:>\n{output_base}', red))
    print(colored(f'[tune model]:>\n{output_tune}', green))
    print('*'*100)
    print('\n')

