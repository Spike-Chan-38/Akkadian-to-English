# =========================
# Deep Past Challenge Starter Notebook
# =========================

# -------------------------
# 0️⃣ Imports
# -------------------------
import pandas as pd
import numpy as np
import re
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments
)
import torch

# -------------------------
# 1️⃣ Load Data
# -------------------------
# Training data
train_df = pd.read_csv("train.csv")
sentences_df = pd.read_csv("Sentences_Oare_FirstWord_LinNum.csv")

# Example test data
test_df = pd.read_csv("test.csv")

print(f"Train data shape: {train_df.shape}")
print(f"Sentences data shape: {sentences_df.shape}")
train_df.head()

# -------------------------
# 2️⃣ Sentence-Level Alignment
# -------------------------
# Merge transliterations with sentence-level info
# (this assumes `oare_id` links train_df and sentences_df)
data = pd.merge(sentences_df, train_df, on="oare_id", how="left")

# Create sentence-level dataset
sentence_data = data[['sentence_transliteration', 'sentence_translation']].dropna()
sentence_data.rename(columns={'sentence_transliteration': 'source', 
                              'sentence_translation': 'target'}, inplace=True)

print(f"Sentence-level data shape: {sentence_data.shape}")
sentence_data.head()

# -------------------------
# 3️⃣ Preprocessing
# -------------------------
def normalize_text(text):
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Optional: normalize apostrophes or rare symbols
    text = re.sub(r"[‘’`]", "'", text)
    text = text.strip()
    return text

sentence_data['source'] = sentence_data['source'].apply(normalize_text)
sentence_data['target'] = sentence_data['target'].apply(normalize_text)

# -------------------------
# 4️⃣ Dataset Preparation for HuggingFace
# -------------------------
dataset = Dataset.from_pandas(sentence_data)
dataset = dataset.train_test_split(test_size=0.1)
print(dataset)

# -------------------------
# 5️⃣ Tokenization
# -------------------------
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_fn(batch):
    source = tokenizer(batch['source'], padding="max_length", truncation=True, max_length=128)
    target = tokenizer(batch['target'], padding="max_length", truncation=True, max_length=128)
    source['labels'] = target['input_ids']
    return source

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# -------------------------
# 6️⃣ Model Setup
# -------------------------
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_steps=50,
    save_total_limit=2,
    evaluation_strategy="steps",
    save_steps=100,
    eval_steps=100,
    weight_decay=0.01,
    predict_with_generate=True,
    fp16=False,
)

# -------------------------
# 7️⃣ Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
)

# -------------------------
# 8️⃣ Train Model
# -------------------------
trainer.train()

# -------------------------
# 9️⃣ Inference Example
# -------------------------
example_sentence = "nu-um-ma e2-ša3"
inputs = tokenizer(example_sentence, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print("Translation:", tokenizer.decode(outputs[0], skip_special_tokens=True))
