from pathlib import Path
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import json
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoTokenizer, AutoModel
import pandas as pd

PROJECT_ROOT = Path(os.environ.get("WRIME_EXP_ROOT", ".")).resolve()

CSV_PATH = str(PROJECT_ROOT / "data" / "processed" / "t_eval_manual" / "t.csv")
CKPT_PATH = str(PROJECT_ROOT / "outputs" / "extractor" / "table311_relaxed_span_window_seed42" / "best.pt")
LABEL_MAP_PATH = str(PROJECT_ROOT / "data" / "processed" / "ash_bio_extractor1000" / "label_map.json")
MODEL_ID = "xlm-roberta-base"
OUT_JSONL = str(PROJECT_ROOT / "outputs" / "spred_cache" / "spred_tcsv_aligned_seed42.jsonl")

MAX_SUBWORD_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BIO_TYPES = {"Aspect", "Opinion"}

def bio_to_spans(tags):
    spans = []
    cur_t, s = None, None
    for i, tg in enumerate(tags + ["O"]):
        if tg == "O" or tg is None:
            if cur_t is not None:
                spans.append((cur_t, s, i - 1))
                cur_t, s = None, None
            continue
        if "-" not in tg:
            continue
        pref, typ = tg.split("-", 1)
        if typ not in BIO_TYPES:
            continue
        if pref == "B" or cur_t != typ:
            if cur_t is not None:
                spans.append((cur_t, s, i - 1))
            cur_t, s = typ, i
    return spans

class XLMRWordCRF(nn.Module):
    def __init__(self, model_id: str, num_tags: int):
        super().__init__()
        self.enc = AutoModel.from_pretrained(model_id)
        hid = self.enc.config.hidden_size
        self.classifier = nn.Linear(hid, num_tags)
        self.crf = CRF(num_tags=num_tags, batch_first=True)

    @torch.no_grad()
    def decode_words(self, input_ids, attention_mask, first_subword_mask_1d):
        hs = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[0]
        word_vecs = hs[first_subword_mask_1d]
        emissions = self.classifier(word_vecs).unsqueeze(0)
        mask = torch.ones((1, emissions.size(1)), dtype=torch.bool, device=emissions.device)
        return self.crf.decode(emissions, mask=mask)[0]

def load_state_dict_compat(model: nn.Module, ckpt_path: str):
    state = torch.load(ckpt_path, map_location=DEVICE)
    sd = state
    if isinstance(state, dict) and "model_state_dict" in state:
        sd = state["model_state_dict"]
    elif isinstance(state, dict) and "model" in state:
        sd = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]

    new_sd = {}
    for k, v in sd.items():
        k2 = k
        if k2.startswith("encoder."):
            k2 = "enc." + k2[len("encoder."):]
        if k2.startswith("fc."):
            k2 = "classifier." + k2[len("fc."):]
        new_sd[k2] = v

    model.load_state_dict(new_sd, strict=False)

def main():
    print("CSV_PATH", CSV_PATH)
    print("OUT_JSONL", OUT_JSONL)

    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)

    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    print("rows", len(df))

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        lm = json.load(f)
    label2id = lm["label2id"]
    id2label = {int(v): k for k, v in label2id.items()}
    num_tags = len(label2id)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("error!error!")

    model = XLMRWordCRF(MODEL_ID, num_tags=num_tags).to(DEVICE)
    load_state_dict_compat(model, CKPT_PATH)
    model.eval()

    wrote = 0
    with open(OUT_JSONL, "w", encoding="utf-8") as w:
        for _, r in df.iterrows():
            ex_id = int(r["id"])
            for which, col in [("o", "tokens_str"), ("p", "perturbed_tokens_str")]:
                words = str(r[col]).split()
                enc = tokenizer(
                    words,
                    is_split_into_words=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_SUBWORD_LEN,
                )
                word_ids = enc.word_ids(batch_index=0)
                enc = enc.to(DEVICE)

                seen = set()
                first_mask = []
                max_wi = -1
                for wi in word_ids:
                    if wi is None:
                        first_mask.append(False)
                    elif wi not in seen:
                        seen.add(wi)
                        first_mask.append(True)
                        if wi > max_wi:
                            max_wi = wi
                    else:
                        first_mask.append(False)
                first_mask = torch.tensor(first_mask, dtype=torch.bool, device=DEVICE)

                n_words = max_wi + 1 if max_wi >= 0 else 0
                words_trunc = words[:n_words]

                pred_ids = model.decode_words(enc["input_ids"], enc["attention_mask"], first_mask)
                pred_tags = [id2label[i] for i in pred_ids]

                spans = bio_to_spans(pred_tags)
                s_pred_sets = [list(range(s, e + 1)) for (_t, s, e) in spans]

                w.write(json.dumps({
                    "id": ex_id,
                    "which": which,
                    "words": words_trunc,
                    "pred_tags": pred_tags,
                    "s_pred_sets": s_pred_sets
                }, ensure_ascii=False) + "\n")
                wrote += 1

    print("wrote", wrote)
    print("saved", OUT_JSONL)

if __name__ == "__main__":
    main()