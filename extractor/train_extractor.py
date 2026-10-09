from pathlib import Path
import os, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from torchcrf import CRF

                
PROJECT_ROOT = Path(os.environ.get("WRIME_EXP_ROOT", ".")).resolve()
DATA_DIR = str(PROJECT_ROOT / "data" / "processed" / "ash_bio_extractor1000")
TRAIN_PATH = os.path.join(DATA_DIR, "train.jsonl")       
DEV_PATH   = os.path.join(DATA_DIR, "dev.jsonl")         
TEST_PATH  = os.path.join(DATA_DIR, "test.jsonl")        
OUT_DIR = str(PROJECT_ROOT / "outputs" / "extractor" / "table311_relaxed_span_window_seed42")
os.makedirs(OUT_DIR, exist_ok=True)

                             
MODEL_ID = "xlm-roberta-base"
MAX_WORDS = 128
MAX_SUB   = 256
BATCH_SIZE = 8
EPOCHS = 5
LR = 3e-5
WEIGHT_DECAY = 0.01
SEED = 42

LABELS = ["O", "B-Aspect", "I-Aspect", "B-Opinion", "I-Opinion"]
O_ID = 0

                     
WINDOW = 1                                           
IOU_TH = 0.0                                       

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

class WordBIO(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        it = self.items[idx]
        words = it["tokens"][:MAX_WORDS]
        labels = it["bio_tag_ids"][:MAX_WORDS]
        return words, labels

def collate_fn(batch, tokenizer):
    words_batch = [b[0] for b in batch]
    labels_batch = [b[1] for b in batch]

    enc = tokenizer(
        words_batch,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_SUB,
        padding="max_length",
        return_tensors="pt",
    )

    B = len(batch)
    first_sub_idx = torch.full((B, MAX_WORDS), -1, dtype=torch.long)
    word_mask = torch.zeros((B, MAX_WORDS), dtype=torch.bool)
    gold_word = torch.full((B, MAX_WORDS), O_ID, dtype=torch.long)

    for b in range(B):
        word_ids = enc.word_ids(batch_index=b)
        prev = None
        wpos = -1
        for si, wid in enumerate(word_ids):
            if wid is None:
                prev = None
                continue
            if wid != prev:
                wpos += 1
                if wpos >= MAX_WORDS:
                    break
                first_sub_idx[b, wpos] = si
                word_mask[b, wpos] = True
            prev = wid

        Lw = min(len(labels_batch[b]), MAX_WORDS)
        if Lw > 0:
            gold_word[b, :Lw] = torch.tensor(labels_batch[b][:Lw], dtype=torch.long)

    return enc["input_ids"], enc["attention_mask"].bool(), first_sub_idx, word_mask, gold_word

class XLMR_WordCRF(nn.Module):
    def __init__(self, model_id, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def gather_word_states(self, last_hidden, first_sub_idx):
        B, _, H = last_hidden.shape
        idx = first_sub_idx.clone()
        idx[idx < 0] = 0
        idx = idx.unsqueeze(-1).expand(B, MAX_WORDS, H)
        return torch.gather(last_hidden, dim=1, index=idx)           

    def forward(self, input_ids, attention_mask, first_sub_idx, word_mask, gold_word=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = self.dropout(out.last_hidden_state)                 
        wh = self.gather_word_states(h, first_sub_idx)          
        emissions = self.fc(wh)                                 
        if gold_word is None:
            return emissions
        loss = -self.crf(emissions, gold_word, mask=word_mask, reduction="mean")
        return loss, emissions

    @torch.no_grad()
    def decode(self, emissions, word_mask):
        return self.crf.decode(emissions, mask=word_mask)

                                                           
def tags_to_spans(tag_ids):
    


       
    spans = []
    n = len(tag_ids)
    i = 0

    def typ_of(t):
        if t in (1,2): return "A"
        if t in (3,4): return "O"
        return None

    while i < n:
        t = int(tag_ids[i])
        typ = typ_of(t)
        if typ is None:
            i += 1
            continue
        start = i
        i += 1
        while i < n and typ_of(tag_ids[i]) == typ:
            i += 1
        end = i - 1
        spans.append((typ, start, end))
    return spans

def expand_span(start, end, L, window):
    s = max(0, start - window)
    e = min(L-1, end + window)
    return s, e

def span_iou(a, b):
                                
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    if inter <= 0:
        return 0.0
    union = (a[1]-a[0]+1) + (b[1]-b[0]+1) - inter
    return inter / max(union, 1)

def relaxed_match_prf(pred_spans, gold_spans, L, window=1, iou_th=0.0):
    





       
            
    pred_by = {"A": [], "O": []}
    gold_by = {"A": [], "O": []}
    for typ, s, e in pred_spans:
        pred_by[typ].append((s, e))
    for typ, s, e in gold_spans:
        gold_by[typ].append((s, e))

    TP = FP = FN = 0

    for typ in ("A", "O"):
        preds = pred_by[typ]
        golds = gold_by[typ]

        used = [False]*len(golds)
        for ps, pe in preds:
            ps2, pe2 = expand_span(ps, pe, L, window)
            best_j = -1
            best_iou = 0.0
            for j, (gs, ge) in enumerate(golds):
                if used[j]:
                    continue
                gs2, ge2 = expand_span(gs, ge, L, window)
                iou = span_iou((ps2, pe2), (gs2, ge2))
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j != -1 and best_iou >= iou_th and best_iou > 0.0:
                TP += 1
                used[best_j] = True
            else:
                FP += 1

        FN += sum(1 for u in used if not u)

    P = TP/(TP+FP+1e-12)
    R = TP/(TP+FN+1e-12)
    F1 = 2*P*R/(P+R+1e-12)
    return P, R, F1

@torch.no_grad()
def eval_relaxed_span(model, loader, window=1, iou_th=0.0):
    model.eval()
    P_list, R_list, F_list = [], [], []

                                                                                   
                                                                     
                           
    TP = FP = FN = 0

    for batch in loader:
        input_ids, attn_mask, first_sub_idx, word_mask, gold_word = batch
        input_ids = input_ids.to(DEVICE)
        attn_mask = attn_mask.to(DEVICE)
        first_sub_idx = first_sub_idx.to(DEVICE)
        word_mask = word_mask.to(DEVICE)
        gold_word = gold_word.to(DEVICE)

        emissions = model(input_ids, attn_mask, first_sub_idx, word_mask)
        paths = model.decode(emissions, word_mask)

        B = gold_word.size(0)
        for b in range(B):
            Lw = int(word_mask[b].sum().item())
            pred_seq = paths[b][:Lw]
            gold_seq = gold_word[b, :Lw].detach().cpu().tolist()

            pred_sp = tags_to_spans(pred_seq)
            gold_sp = tags_to_spans(gold_seq)

                                                                
                                            
                                                                      
                                    
                                        
                           
                      

                 
            pred_by = {"A": [], "O": []}
            gold_by = {"A": [], "O": []}
            for typ, s, e in pred_sp:
                pred_by[typ].append((s, e))
            for typ, s, e in gold_sp:
                gold_by[typ].append((s, e))

            for typ in ("A","O"):
                preds = pred_by[typ]
                golds = gold_by[typ]
                used = [False]*len(golds)
                for ps, pe in preds:
                    ps2, pe2 = expand_span(ps, pe, Lw, window)
                    best_j = -1
                    best_iou = 0.0
                    for j,(gs,ge) in enumerate(golds):
                        if used[j]:
                            continue
                        gs2, ge2 = expand_span(gs, ge, Lw, window)
                        iou = span_iou((ps2,pe2),(gs2,ge2))
                        if iou > best_iou:
                            best_iou = iou
                            best_j = j
                    if best_j != -1 and best_iou >= iou_th and best_iou > 0.0:
                        TP += 1
                        used[best_j] = True
                    else:
                        FP += 1
                FN += sum(1 for u in used if not u)

    P = TP/(TP+FP+1e-12)
    R = TP/(TP+FN+1e-12)
    F1 = 2*P*R/(P+R+1e-12)
    return P, R, F1

def main():
    set_seed(SEED)
    train_items = load_jsonl(TRAIN_PATH)
    dev_items   = load_jsonl(DEV_PATH)
    test_items  = load_jsonl(TEST_PATH)
    print("train_n", len(train_items), "dev_n", len(dev_items), "test_n", len(test_items))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    train_ds = WordBIO(train_items)
    dev_ds   = WordBIO(dev_items)
    test_ds  = WordBIO(test_items)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, tokenizer))
    dev_loader   = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=lambda b: collate_fn(b, tokenizer))
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=lambda b: collate_fn(b, tokenizer))

    model = XLMR_WordCRF(MODEL_ID, num_labels=len(LABELS)).to(DEVICE)
    optim = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_dev_f1 = -1.0
    best_path = os.path.join(OUT_DIR, "best.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids, attn_mask, first_sub_idx, word_mask, gold_word = batch
            input_ids = input_ids.to(DEVICE)
            attn_mask = attn_mask.to(DEVICE)
            first_sub_idx = first_sub_idx.to(DEVICE)
            word_mask = word_mask.to(DEVICE)
            gold_word = gold_word.to(DEVICE)

            optim.zero_grad()
            loss, _ = model(input_ids, attn_mask, first_sub_idx, word_mask, gold_word=gold_word)
            loss.backward()
            optim.step()
            total_loss += float(loss.item())

        dev_p, dev_r, dev_f1 = eval_relaxed_span(model, dev_loader, window=WINDOW, iou_th=IOU_TH)

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            torch.save({"model": model.state_dict(), "model_id": MODEL_ID, "labels": LABELS}, best_path)

    print("best_path", best_path)
    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])

    test_p, test_r, test_f1 = eval_relaxed_span(model, test_loader, window=WINDOW, iou_th=IOU_TH)
    print("precision", test_p)
    print("recall", test_r)
    print("f1", test_f1)
    print("saved", best_path)

if __name__ == "__main__":
    main()