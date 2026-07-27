from configs.configs import *
from functools import partial
import warnings
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM , TrainerCallback , default_data_collator , Trainer, TrainingArguments , set_seed , logging , AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from datasets import Dataset , concatenate_datasets
from transformers import BitsAndBytesConfig
import numpy as np
from huggingface_hub import login , upload_folder
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainerCallback

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it", trust_remote_code=True )
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it", trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)

dataset = pd.read_csv("Dial2MSA_correct.csv" , encoding = 'utf-8-sig')

instruction = "Translate from Modern Standard Arabic into Egyptian Dialect, Output only the translation, do NOT output anything else before nor after it."

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def preprocess(examples, max_length=256):
    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for msa_sentence, target in zip(examples["msa"], examples["Egyptian"]):
        msa_sentence = str(msa_sentence) if msa_sentence is not None else ""
        target = str(target) if target is not None else ""

        full_prompt = instruction + "\n" + msa_sentence

        messages = [{"role": "user", "content": full_prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        target_text = target + tokenizer.eos_token

        max_prompt_len = max_length - 32
        prompt_ids = tokenizer(
            prompt_text, 
            add_special_tokens=False, 
            truncation=True, 
            max_length=max_prompt_len, 
            return_tensors="pt"
        )["input_ids"][0]

        max_target_length = max_length - len(prompt_ids)
        target_ids = tokenizer(
            target_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_target_length,
            return_tensors="pt"
        )["input_ids"][0]

        if len(prompt_ids) + len(target_ids) > max_length:
            prompt_ids = prompt_ids[: max_length - 1]
            target_ids = target_ids[:1]  

        input_ids = prompt_ids.tolist() + target_ids.tolist()
        labels = [-100] * len(prompt_ids) + target_ids.tolist()
        attention_mask = [1] * len(input_ids)

        padding_length = max_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        labels += [-100] * padding_length
        attention_mask += [0] * padding_length

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_mask_list,
        "token_type_ids": [[0]*len(ids) for ids in input_ids_list]
    }

tokenized_train = dataset.map(preprocess , batched = True)

data_collator = default_data_collator

training_args = TrainingArguments(
    output_dir = OUTPUT_DIR,
    save_strategy="steps",
    save_steps= SAVE_STEPS,
    seed = SEED,
    learning_rate= lr,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps= GRAD_ACCUM,
    weight_decay=WEIGHT_DECAY,
    save_total_limit=2,
    num_train_epochs=NUM_EPOCHS,
    fp16=False,
    bf16=True,
    logging_steps=LOGGING_STEPS,
    report_to="wandb",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    #tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer.train()

'''
checkpoint = ''
torch.load = partial(torch.load, weights_only=False)
trainer.train(resume_from_checkpoint = checkpoint)
'''

trainer.save_model(full_model_path)

upload_folder(folder_path=full_model_path, repo_id= repo_id , repo_type="model",  token = model_token)