from pathlib import Path



import json
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import re

import pandas as pd
import torch
from fugashi import Tagger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(os.environ.get("WRIME_EXP_ROOT", ".")).resolve()

CSV_PATH = str(PROJECT_ROOT / "data" / "processed" / "wrime_subjective_binary" / "test.csv")
ROOT_DIR = str(PROJECT_ROOT / "outputs" / "baseline" / "base_seed42")

MAX_LEN = 128
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
tagger = Tagger()


def wakachi_space(text: str) -> str:
    return " ".join(word.surface for word in tagger(str(text)))


def has_weights(directory: str) -> bool:
    if not os.path.isdir(directory):
        return False
    if not os.path.isfile(os.path.join(directory, "config.json")):
        return False
    return any(os.path.isfile(os.path.join(directory, name)) for name in WEIGHT_FILES)


def pick_best_dir(root_dir: str) -> str:
    trainer_state = os.path.join(root_dir, "trainer_state.json")
    if os.path.isfile(trainer_state):
        try:
            with open(trainer_state, "r", encoding="utf-8") as handle:
                state = json.load(handle)
            best = state.get("best_model_checkpoint")
            if best:
                if not os.path.isabs(best):
                    best = os.path.join(root_dir, best)
                if has_weights(best):
                    return best
        except (OSError, ValueError, TypeError):
            pass

    candidates = []
    for name in os.listdir(root_dir):
        directory = os.path.join(root_dir, name)
        if os.path.isdir(directory) and re.fullmatch(r"checkpoint-\d+", name) and has_weights(directory):
            candidates.append((int(name.rsplit("-", 1)[1]), directory))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]

    if has_weights(root_dir):
        return root_dir

    raise FileNotFoundError("error!error!")


@torch.no_grad()
def evaluate_accuracy(texts, labels, tokenizer, model):
    model.eval()
    correct = 0
    total = 0
    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start:start + BATCH_SIZE]
        batch_labels = torch.tensor(
            labels[start:start + BATCH_SIZE],
            dtype=torch.long,
            device=DEVICE,
        )
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
        predictions = model(**encoded).logits.argmax(dim=-1)
        correct += (predictions == batch_labels).sum().item()
        total += batch_labels.size(0)
    return correct / max(total, 1)


def main():
    if not os.path.isdir(ROOT_DIR):
        raise FileNotFoundError("error!error!")

    dataframe = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    for column in ("text", "label"):
        if column not in dataframe.columns:
            raise ValueError("error!error!")
    dataframe["text"] = dataframe["text"].astype(str)
    dataframe["label"] = dataframe["label"].astype(int)

    model_dir = pick_best_dir(ROOT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=False,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    ).to(DEVICE)

    texts = [wakachi_space(text) for text in dataframe["text"].tolist()]
    labels = dataframe["label"].tolist()
    accuracy = evaluate_accuracy(texts, labels, tokenizer, model)

    print("model", "roberta_base", "seed", 42, "acc_o", accuracy, "n", len(dataframe), "model_dir", model_dir)


if __name__ == "__main__":
    main()
