import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, GenerationConfig
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer
import warnings
warnings.filterwarnings("ignore")



# Loading original model
model_name = "tiiuae/falcon-7b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                       # 表示以4位精度加载模型
    bnb_4bit_quant_type="nf4",               # 预训练模型应该以4位NF格式进行量化
    bnb_4bit_use_double_quant=True,          # 采用QLoRA论文中提到的双量化
    bnb_4bit_compute_dtype=torch.float16,    # 在计算过程中，预训练模型应以BF16格式加载
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


# Loading PEFT model
PEFT_MODEL = "./falcon-7b-sharded-bf16-finetuned-mental-health-conversational/checkpoint-320"

config = PeftConfig.from_pretrained(PEFT_MODEL)
peft_base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

peft_model = PeftModel.from_pretrained(peft_base_model, PEFT_MODEL)

peft_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
peft_tokenizer.pad_token = peft_tokenizer.eos_token


# Function to generate responses from both original model and PEFT model and compare their answers.
def generate_answer(query):
    system_prompt = """Answer the following question truthfully.
    If you don't know the answer, respond 'Sorry, I don't know the answer to this question.'.
    If the question is too complex, respond 'Kindly, consult a psychiatrist for further queries.'."""

    user_prompt = f"""
    <HUMAN>: {query}
    <ASSISTANT>: 
    """

    final_prompt = system_prompt + "\n" + user_prompt

    device = "cuda:0"
    dashline = "-".join("" for i in range(50))

    encoding = tokenizer(final_prompt, return_tensors="pt").to(device)
    outputs = model.generate(input_ids=encoding.input_ids, generation_config=GenerationConfig(max_new_tokens=256, pad_token_id = tokenizer.eos_token_id, \
                                                                                                                        eos_token_id = tokenizer.eos_token_id, attention_mask = encoding.attention_mask, \
                                                                                                                        temperature=0.4, top_p=0.6, repetition_penalty=1.3, num_return_sequences=1,))
    # text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_ids = outputs[0]
    generated_ids = generated_ids[len(encoding.input_ids[0]):]
    text_output = tokenizer.decode(generated_ids, skip_special_tokens=True)


    print(dashline)
    print(f'ORIGINAL MODEL RESPONSE:\n{text_output}')
    print(dashline)

    peft_encoding = peft_tokenizer(final_prompt, return_tensors="pt").to(device)
    peft_outputs = peft_model.generate(input_ids=peft_encoding.input_ids, generation_config=GenerationConfig(max_new_tokens=256, pad_token_id = peft_tokenizer.eos_token_id, \
                                                                                                                        eos_token_id = peft_tokenizer.eos_token_id, attention_mask = peft_encoding.attention_mask, \
                                                                                                                        temperature=0.4, top_p=0.6, repetition_penalty=1.3, num_return_sequences=1,))
    # peft_text_output = peft_tokenizer.decode(peft_outputs[0], skip_special_tokens=True)
    peft_generated_ids = peft_outputs[0]
    peft_generated_ids = peft_generated_ids[len(encoding.input_ids[0]):]
    peft_text_output = peft_tokenizer.decode(peft_generated_ids, skip_special_tokens=True)

    print(f'PEFT MODEL RESPONSE:\n{peft_text_output}')
    print(dashline)


# query = "How can I prevent anxiety and depression?"
# generate_answer(query)

while True:
    query = input("enter your question: ")
    generate_answer(query)



