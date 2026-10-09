


import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from fugashi import Tagger
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(os.environ.get("WRIME_EXP_ROOT", ".")).resolve()

DEFAULT_CSV_PATH = str(PROJECT_ROOT / "data" / "processed" / "wrime_subjective_binary" / "test.csv")
DEFAULT_CKPT_PATH = str(PROJECT_ROOT / "outputs" / "baseline" / "bilstm_seed42" / "best.pt")

MAX_LEN = 128
BATCH_SIZE = 32
PAD = "<pad>"
UNK = "<unk>"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tagger = Tagger()


def tokenize(text: str):
    return [word.surface for word in tagger(str(text))]


def encode(tokens, stoi, max_len):
    unk_id = stoi.get(UNK, 1)
    pad_id = stoi.get(PAD, 0)
    ids = [stoi.get(token, unk_id) for token in tokens[:max_len]]
    mask = [True] * len(ids)
    pad_count = max_len - len(ids)
    if pad_count > 0:
        ids.extend([pad_id] * pad_count)
        mask.extend([False] * pad_count)
    return ids, mask


class TextDataset(Dataset):
    def __init__(self, dataframe, stoi):
        self.texts = dataframe["text"].astype(str).tolist()
        self.labels = dataframe["label"].astype(int).tolist()
        self.stoi = stoi

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        ids, mask = encode(tokenize(self.texts[index]), self.stoi, MAX_LEN)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden=196, dropout=0.2, num_labels=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.attn_fc = nn.Linear(hidden * 2, 1)
        self.classifier = nn.Linear(hidden * 2, num_labels)

    def forward(self, input_ids, attention_mask):
        hidden, _ = self.lstm(self.emb(input_ids))
        hidden = self.dropout(hidden)
        scores = self.attn_fc(hidden).squeeze(-1)
        scores = scores.masked_fill(~attention_mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(hidden * weights.unsqueeze(-1), dim=1)
        logits = self.classifier(self.dropout(pooled))
        return logits, weights


@torch.no_grad()
def evaluate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        labels = labels.to(DEVICE)
        logits, _ = model(input_ids, attention_mask)
        predictions = logits.argmax(dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT_PATH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()
    if not Path(args.csv_path).is_file():
        raise FileNotFoundError("error!error!")
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError("error!error!")

    dataframe = pd.read_csv(args.csv_path, encoding="utf-8-sig")
    for column in ("text", "label"):
        if column not in dataframe.columns:
            raise ValueError("error!error!")
    dataframe["text"] = dataframe["text"].astype(str)
    dataframe["label"] = dataframe["label"].astype(int)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "stoi" not in checkpoint or "model" not in checkpoint:
        raise KeyError("error!error!")

    stoi = checkpoint["stoi"]
    model = BiLSTMAttention(len(stoi)).to(DEVICE)
    model.load_state_dict(checkpoint["model"], strict=True)

    dataset = TextDataset(dataframe, stoi)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    accuracy = evaluate_accuracy(model, loader)

    print("model", "bilstm", "seed", 42, "acc_o", accuracy, "n", len(dataframe))


if __name__ == "__main__":
    main()
