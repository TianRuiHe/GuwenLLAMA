import random

import transformers
import textwrap
from transformers import LlamaTokenizer, LlamaForCausalLM
import os
import sys
from typing import List

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)

import fire
import torch
from datasets import load_dataset
from datasets import Dataset
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pylab import rcParams
from torch.optim.lr_scheduler import LambdaLR

BASE_MODEL = "../BaseoModel"
CUTOFF_LEN = 1024
train_on_inputs=False
REMOVE_MARK_RATIO = 0.5
BATCH_SIZE = 48
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-5
TRAIN_STEPS = -1
TRAIN_EPOCH = 3
WARMUP_STEPS = 50
LR_SCHEDULER="cosine"
WARMUP_RATIO = 0.03
MAX_GRAD_NORM = 0.3
GROUP_BY_LENGTH = True
OUTPUT_DIR = "../SaveModel"




def remove_punctuation(text):
    # 使用正则表达式匹配所有标点符号，并将其替换为空格
    # print(text)
    text=str(text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace(" ","")
    text=str(text)
    # print(text)
    # print("===========================================================================================================================")
    return text


traindata = pd.read_csv("../Data/DataInstruction_merge2__train.csv")
traindata = traindata.sample(frac=1, random_state=42)
valdata = pd.read_csv("../Data/DataInstruction_merge2__val.csv")
valdata = valdata.sample(frac=1, random_state=42)[0:1000]

# BASE_MODEL = "/home/htr/Works/LargeLanguageModel/OriModel/Chinese_Alpaca_Plus_13B_huggingface"


model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"


def generate_prompt(data_point , remove_mark_ratio , random_number):
    instruction = data_point["instruction"]
    input = data_point["input"]
    output = data_point["output"]
    if random_number<remove_mark_ratio:
        input = remove_punctuation(input)
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501
### Instruction:
{instruction}
### Input:
{input}
### Response:
{output}"""
def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < CUTOFF_LEN
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result
def generate_and_tokenize_prompt(data_point , add_eos_token=True, remove_mark_ratio = REMOVE_MARK_RATIO):
    if remove_mark_ratio >= 0:
        random_number = random.random()
    else:
        random_number = float('inf')
    full_prompt = generate_prompt(data_point , remove_mark_ratio , random_number)
    tokenized_full_prompt = tokenize(full_prompt)
    print(full_prompt)
    print("======================================================================================================")
    if  train_on_inputs==False:
        instruction = data_point["instruction"]
        input = data_point["input"]
        if random_number<remove_mark_ratio:
            input = remove_punctuation(input)
        prompt_instruction_input = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501
        ### Instruction:
        {instruction}
        ### Input:
        {input}"""

        user_prompt = prompt_instruction_input
        tokenized_user_prompt = tokenize(
            user_prompt
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        if add_eos_token:
            user_prompt_len -= 1

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_prompt


train_dictdata = Dataset.from_pandas(traindata)
val_dictdata = Dataset.from_pandas(valdata)
train_data = (
    train_dictdata.map(generate_and_tokenize_prompt)
)
val_data = (
    val_dictdata.map(generate_and_tokenize_prompt)
)


LORA_R = 128
LORA_ALPHA = 16
LORA_DROPOUT= 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
    "k_proj",
]




model = prepare_model_for_int8_training(model)
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()


training_arguments = transformers.TrainingArguments(
    num_train_epochs=TRAIN_EPOCH,
    max_steps=TRAIN_STEPS,
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    lr_scheduler_type=LR_SCHEDULER,
    warmup_ratio=WARMUP_RATIO,
    learning_rate=LEARNING_RATE,
    max_grad_norm=MAX_GRAD_NORM,
    group_by_length=GROUP_BY_LENGTH,
    fp16=True,
    logging_steps=30,
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=10,
    save_steps=10,
    output_dir=OUTPUT_DIR,
    save_total_limit=100000000,
    load_best_model_at_end=True,
    report_to="tensorboard"
)


data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_arguments,
    data_collator=data_collator,
)


# a=input("wait")

model.config.use_cache = False
old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(
        self, old_state_dict()
    )
).__get__(model, type(model))

model = torch.compile(model)

trainer.train()
model.save_pretrained(OUTPUT_DIR)


