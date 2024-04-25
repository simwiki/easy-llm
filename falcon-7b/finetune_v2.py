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
wandb.init(mode="offline")


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

        train_data = train_data.select(range(2))
        valid_data = valid_data.select(range(2))
        print(data)
        print(train_data)
        print(valid_data)
        return train_data, valid_data
    except Exception as e:
        print(f"Error in preparing data: {e}, Line: {e.__traceback__.tb_lineno}")
        raise e


model_name = "tiiuae/falcon-7b"
output_dir = "./falcon_7b_lora_v4"
wandb_run_name = f"run-falcon-7b-lora-v4-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"

tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True)
print(f"Using tokenizer: {tokenizer.__class__.__name__}")
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

DATA_PATH = "b-mc2/sql-create-context"
data = load_dataset(DATA_PATH)
train_data, valid_data = prepare_data(data)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # load model in 4-bit precision
    bnb_4bit_quant_type="nf4",              # pre-trained model should be quantized in 4-bit NF format
    bnb_4bit_use_double_quant=True,         # Using double quantization as mentioned in QLoRA paper
    bnb_4bit_compute_dtype=torch.float16,   # During computation, pre-trained model should be loaded in FP16 format
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,         # Use bitsandbytes config
    device_map="auto",                      # Specifying device_map="auto" so that HF Accelerate will determine which GPU to put each layer of the model on
    trust_remote_code=True,                # Set trust_remote_code=True to use falcon-7b model with custom code
)
# model.print_trainable_parameters()          # Be more transparent about the % of trainable params.

lora_config = {
    'r': 32,                                # dimension of the low-rank matrices
    'lora_alpha' : 16,                      # scaling factor for the weight matrices
    'lora_dropout' : 0.1,                   # dropout probability of the LoRA layers
    'bias': 'none',                         # setting to 'none' for only training weight params instead of biases
    'task_type': 'CAUSAL_LM',
    'target_modules': [                     # Setting names of modules in falcon-7b model that we want to apply LoRA to
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]
}
peft_config = LoraConfig(**lora_config)
model = get_peft_model(model, peft_config)

trainer = Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=valid_data,
    args=TrainingArguments(
        # lr_scheduler_type= "cosine",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        max_steps=200,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=20,
        save_steps=20,
        save_total_limit=5,
        output_dir=output_dir,
        load_best_model_at_end=False,
        group_by_length=True,                               # faster, but produces an odd training loss curve
        report_to="wandb",
        run_name=wandb_run_name,
    ),
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
)
model.config.use_cache = False

print( model.state_dict)
trainer.train(resume_from_checkpoint=None)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

wandb.finish()
print("=> Trained the model successfully")
del trainer
torch.cuda.synchronize()


