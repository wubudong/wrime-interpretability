import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from fugashi import Tagger


DATA_DIR = "data/processed/wrime_subjective_binary"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
DEV_PATH   = os.path.join(DATA_DIR, "dev.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test.csv")

OUT_DIR = "outputs/baseline/bilstm_seed42"
os.makedirs(OUT_DIR, exist_ok=True)


MAX_LEN = 128
BATCH_SIZE = 32
EMB_DIM = 128
HIDDEN = 196
DROPOUT = 0.2
LR = 5e-5
EPOCHS = 5
SEED = 42
PAD = "<pad>"
UNK = "<unk>"

tagger = Tagger()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def tokenize(text: str):

    return [w.surface for w in tagger(str(text))]

def build_vocab(token_lists, min_freq=2):
    from collections import Counter
    c = Counter()
    for toks in token_lists:
        c.update(toks)
    stoi = {PAD: 0, UNK: 1}
    for w, f in c.items():
        if f >= min_freq and w not in stoi:
            stoi[w] = len(stoi)
    return stoi

def encode(tokens, stoi, max_len):
    ids = [stoi.get(t, stoi[UNK]) for t in tokens][:max_len]
    attn = [1]*len(ids)

    if len(ids) < max_len:
        pad_n = max_len - len(ids)
        ids += [stoi[PAD]] * pad_n
        attn += [0] * pad_n
    return ids, attn

class TxtDS(Dataset):
    def __init__(self, df, stoi, max_len):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.stoi = stoi
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        toks = tokenize(self.texts[idx])
        ids, attn = encode(toks, self.stoi, self.max_len)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(attn, dtype=torch.bool),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden=196, dropout=0.2, num_labels=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

        self.attn_fc = nn.Linear(hidden * 2, 1)
        self.classifier = nn.Linear(hidden * 2, num_labels)

    def forward(self, input_ids, attention_mask):
        x = self.emb(input_ids)
        x, _ = self.lstm(x)
        x = self.dropout(x)

        scores = self.attn_fc(x).squeeze(-1)

        scores = scores.masked_fill(~attention_mask, -1e9)
        weights = torch.softmax(scores, dim=1)

        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        logits = self.classifier(self.dropout(pooled))
        return logits

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for ids, attn, y in loader:
        ids = ids.to(device)
        attn = attn.to(device)
        y = y.to(device)
        logits = model(ids, attn)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    train_df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    dev_df   = pd.read_csv(DEV_PATH, encoding="utf-8-sig")
    test_df  = pd.read_csv(TEST_PATH, encoding="utf-8-sig")

    print("train_n", len(train_df), "dev_n", len(dev_df), "test_n", len(test_df))
    print("train_label_counts", train_df["label"].value_counts())


    train_tokens = [tokenize(t) for t in train_df["text"].astype(str).tolist()]
    stoi = build_vocab(train_tokens, min_freq=1)
    print("vocab_size", len(stoi))

    train_ds = TxtDS(train_df, stoi, MAX_LEN)
    dev_ds   = TxtDS(dev_df, stoi, MAX_LEN)
    test_ds  = TxtDS(test_df, stoi, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dev_loader   = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = BiLSTMAttention(vocab_size=len(stoi), emb_dim=EMB_DIM, hidden=HIDDEN, dropout=DROPOUT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    best_dev = 0.0
    best_path = os.path.join(OUT_DIR, "best.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for ids, attn, y in train_loader:
            ids = ids.to(device)
            attn = attn.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits = model(ids, attn)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        dev_acc = evaluate(model, dev_loader, device)

        if dev_acc > best_dev:
            best_dev = dev_acc
            torch.save({"model": model.state_dict(), "stoi": stoi}, best_path)


    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_acc = evaluate(model, test_loader, device)
    print("best_dev_acc", best_dev)
    print("test_acc", test_acc)
    print("saved", best_path)

if __name__ == "__main__":
    main()