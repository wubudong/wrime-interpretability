

import os
import pandas as pd
import numpy as np

from datasets import Dataset
from fugashi import Tagger

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate


DATA_DIR = "data/processed/wrime_subjective_binary"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
DEV_PATH   = os.path.join(DATA_DIR, "dev.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test.csv")


MODEL_ID = "nlp-waseda/roberta-large-japanese"
OUT_DIR = "outputs/baseline/large_seed42"


MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 3
LR = 1e-5
SEED = 42
acc_metric = evaluate.load("accuracy")
tagger = Tagger()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return acc_metric.compute(predictions=preds, references=labels)

def df_to_hf_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df[["text", "label"]].copy())

def wakachi_space(text: str) -> str:

    return " ".join([w.surface for w in tagger(text)])

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    dev_df   = pd.read_csv(DEV_PATH, encoding="utf-8-sig")
    test_df  = pd.read_csv(TEST_PATH, encoding="utf-8-sig")

    print("train_n", len(train_df), "dev_n", len(dev_df), "test_n", len(test_df))
    print("train_label_counts", train_df["label"].value_counts())
    print("dev_label_counts", dev_df["label"].value_counts())
    print("test_label_counts", test_df["label"].value_counts())

    train_ds = df_to_hf_dataset(train_df)
    dev_ds   = df_to_hf_dataset(dev_df)
    test_ds  = df_to_hf_dataset(test_df)


    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

    def tokenize_batch(batch):
        texts = [wakachi_space(t) for t in batch["text"]]
        return tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
        )

    train_ds = train_ds.map(tokenize_batch, batched=True)
    dev_ds   = dev_ds.map(tokenize_batch, batched=True)
    test_ds  = test_ds.map(tokenize_batch, batched=True)

    cols = ["input_ids", "attention_mask", "label"]
    train_ds.set_format(type="torch", columns=cols)
    dev_ds.set_format(type="torch", columns=cols)
    test_ds.set_format(type="torch", columns=cols)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)

    args = TrainingArguments(
        output_dir=OUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="no",
        disable_tqdm=True,
        learning_rate=LR,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        seed=SEED,
        report_to="none",
        bf16=True,

    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_result = trainer.evaluate(test_ds)
    print("test_result", test_result)


    trainer.save_model(OUT_DIR)
    print("saved", OUT_DIR)

if __name__ == "__main__":
       main()