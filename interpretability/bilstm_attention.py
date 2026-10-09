from pathlib import Path











import os
import re
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fugashi import Tagger




PROJECT_ROOT = Path(os.environ.get("WRIME_EXP_ROOT", ".")).resolve()

MODEL_PATH = str(PROJECT_ROOT / "outputs" / "baseline" / "bilstm_seed42" / "best.pt")
CSV_PATH   = str(PROJECT_ROOT / "data" / "processed" / "t_eval_manual" / "t.csv")


S_PRED_JSONL = str(PROJECT_ROOT / "outputs" / "spred_cache" / "spred_tcsv_paired_mark_unpaired_seed42.jsonl")



PN_TABLE_PATH = str(PROJECT_ROOT / "data" / "raw3" / "pn_ja.dic.txt")
PN_ALPHA = 0.5


S_DILATE_WINDOW = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 128
H_CONST = 0.5
RUN_SEED = 42
MODEL_NAME = "bilstm"
EXPLAINER_NAME = "att"
RESULTS_DIR = str(PROJECT_ROOT / "outputs" / "table4_per_instance" / f"seed{RUN_SEED}")
RESULT_CSV = os.path.join(
    RESULTS_DIR, f"{MODEL_NAME}_{EXPLAINER_NAME}_seed{RUN_SEED}_per_instance.csv"
)
UNK = "<unk>"




DEBUG = True
DEBUG_MAX_EX = 20
DEBUG_WINDOWS = [0, 1, 2, 3 ]
DEBUG_SEED = 42
DEBUG_SAMPLE_PROB = 0.05




class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden=196, dropout=0.2, num_labels=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.attn_fc = nn.Linear(hidden * 2, 1)
        self.classifier = nn.Linear(hidden * 2, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        x = self.emb(input_ids)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        scores = self.attn_fc(x).squeeze(-1)
        scores = scores.masked_fill(~attention_mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        logits = self.classifier(self.dropout(pooled))
        return logits, weights




_span_num_re = re.compile(r"(\d+)(?:\s*-\s*(\d+))?")

def parse_one_span_part(part: str):
    part = str(part).strip()
    if not part:
        return set()
    num_part = part.split("(")[0].strip()
    m = _span_num_re.search(num_part)
    if not m:
        return set()
    a = int(m.group(1))
    b = int(m.group(2)) if m.group(2) else a
    return set(range(a - 1, b))

def parse_span_str_to_slots(span_str):
    if span_str is None or (
        isinstance(span_str, float) and pd.isna(span_str)
    ):
        return [set()]

    s = str(span_str).strip()
    if not s or s.lower() == "(none)":
        return [set()]

    return [parse_one_span_part(part) for part in s.split("|")]


def build_gold_sets_paired(row, actual_len: int):
    aspect_slots = parse_span_str_to_slots(
        row.get("aspect_spans", "")
    )
    opinion_slots = parse_span_str_to_slots(
        row.get("opinion_spans", "")
    )

    slot_count = max(len(aspect_slots), len(opinion_slots))
    gold_sets = []

    for i in range(slot_count):
        aspect_set = (
            aspect_slots[i] if i < len(aspect_slots) else set()
        )
        opinion_set = (
            opinion_slots[i] if i < len(opinion_slots) else set()
        )

        gold_set = aspect_set | opinion_set
        gold_set = {
            index
            for index in gold_set
            if 0 <= index < actual_len
        }

        if gold_set:
            gold_sets.append(gold_set)

    return gold_sets

def union_rationales(gold_sets):
    u = set()
    for s in gold_sets:
        u |= set(s)
    return u




def token_f1_wang(pred_set, gold_sets):
    if not gold_sets:
        return 0.0
    pred_set = set(pred_set)
    if len(pred_set) == 0:
        return 0.0
    best = 0.0
    for g in gold_sets:
        g = set(g)
        if len(g) == 0:
            continue
        inter = len(pred_set & g)
        if inter == 0:
            continue
        p = inter / len(pred_set)
        r = inter / len(g)
        f1 = 2 * p * r / (p + r)
        best = max(best, f1)
    return best

def compute_ranking_map(Xo_sorted, Xp_sorted):
    Lp = len(Xp_sorted)
    if Lp == 0:
        return 0.0
    total = 0.0
    for i in range(1, Lp + 1):
        prefix_o = set(Xo_sorted[:min(i, len(Xo_sorted))])
        hits = 0
        for j in range(1, i + 1):
            hits += 1 if Xp_sorted[j - 1] in prefix_o else 0
        total += hits / i
    return total / Lp




def topk_indices(scores: np.ndarray, k: int):
    scores = np.asarray(scores)
    k = max(1, int(k))
    order = np.argsort(-scores, kind="stable")
    return set(map(int, order[:min(k, len(scores))].tolist()))

def sort_tokens_by_scores(tokens, scores):
    scores = np.asarray(scores)
    order = np.argsort(-scores, kind="stable")
    return [tokens[i] for i in order]




def compute_dataset_rlr_mean(df):
    ratios = []
    for _, row in df.iterrows():
        tokens = str(row["tokens_str"]).split()
        if not tokens:
            continue
        actual_len = min(len(tokens), MAX_LEN)
        gold_sets = build_gold_sets_paired(row, actual_len)
        if not gold_sets:
            continue
        r_len = len(union_rationales(gold_sets))
        ratios.append(r_len / actual_len)
    return float(np.mean(ratios)) if ratios else 0.0




def load_spred_sets(path: str):
    mp = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            ex_id = int(obj["id"])
            which = obj["which"]
            sets_all = obj.get("s_pred_sets", [])
            mp[(ex_id, which)] = [set(map(int, s)) for s in sets_all]
    return mp




def apply_weighting(w0: np.ndarray, s_all, h=0.5):
    w0 = np.asarray(w0, dtype=np.float32)
    L = len(w0)
    W_lists = [[float(v)] for v in w0.tolist()]

    for s in s_all:
        ws = 0.0
        for i in s:
            if 0 <= i < L:
                ws += float(w0[i])
        for t in s:
            if 0 <= t < L:
                wt = (1.0 - h) * float(w0[t]) + h * ws
                W_lists[t].append(float(wt))

    return np.array([max(lst) for lst in W_lists], dtype=np.float32)




def apply_weighting_ws_dilate(w0: np.ndarray, s_all, h=0.5, window=3):








    w0 = np.asarray(w0, dtype=np.float32)
    L = len(w0)
    W_lists = [[float(v)] for v in w0.tolist()]

    for s in s_all:
        dil = set()
        for i in s:
            for d in range(-window, window + 1):
                j = i + d
                if 0 <= j < L:
                    dil.add(j)

        ws = float(np.sum(w0[list(dil)])) if dil else 0.0

        for t in s:
            if 0 <= t < L:
                wt = (1.0 - h) * float(w0[t]) + h * ws
                W_lists[t].append(float(wt))

    return np.array([max(lst) for lst in W_lists], dtype=np.float32)




def load_pn_table(path):
    pn_dict = {}
    for enc in ["utf-8", "shift_jis", "cp932"]:
        try:
            with open(path, "r", encoding=enc) as f:
                for line in f:
                    parts = line.strip().split(":")
                    if len(parts) >= 4:
                        pn_dict[parts[0]] = abs(float(parts[3]))
            return pn_dict
        except Exception:
            continue
    return {}

def fuse_att_with_pn(tokens, w0, tagger, pn_dict, pn_max, alpha=0.5):
    w0 = np.asarray(w0, dtype=np.float32)
    pn_scores = np.zeros_like(w0, dtype=np.float32)

    for i, t in enumerate(tokens):
        try:
            node = tagger(t)[0]
            lemma = node.feature.lemma if getattr(node.feature, "lemma", None) else t
        except Exception:
            lemma = t
        pn_val = pn_dict.get(lemma, pn_dict.get(t, 0.0))
        pn_scores[i] = float(pn_val)

    if pn_max > 0:
        pn_scores = pn_scores / float(pn_max)

    fused = (1.0 - alpha) * w0 + alpha * pn_scores
    return fused.astype(np.float32)




def stats_sets(s_all, L):

    if not s_all:
        return 0, 0.0, 0.0
    sizes = []
    cover = set()
    for s in s_all:
        ss = {i for i in s if 0 <= i < L}
        if ss:
            sizes.append(len(ss))
            cover |= ss
    if not sizes:
        return 0, 0.0, 0.0
    return len(sizes), float(np.mean(sizes)), (len(cover) / float(L) if L > 0 else 0.0)








def validate_extended_columns(df):
    required = {
        "perturbed_numbers",
        "perturbed_aspect_spans",
        "perturbed_opinion_spans",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("error!error!")


def build_gold_sets_paired_strict(row, actual_len, aspect_col, opinion_col):

    aspect_slots = parse_span_str_to_slots(row.get(aspect_col, ""))
    opinion_slots = parse_span_str_to_slots(row.get(opinion_col, ""))
    if len(aspect_slots) != len(opinion_slots):
        raise ValueError("error!error!")

    paired_sets = []
    for slot_index, (aspect_set, opinion_set) in enumerate(
        zip(aspect_slots, opinion_slots), start=1
    ):
        merged = {
            int(i) for i in (set(aspect_set) | set(opinion_set))
            if 0 <= int(i) < actual_len
        }
        if not merged:
            raise ValueError("error!error!")
        paired_sets.append(merged)
    return paired_sets


def token_to_lemma(token, tagger):
    try:
        nodes = tagger(str(token))
        if nodes:
            lemma = getattr(nodes[0].feature, "lemma", None)
            if lemma and lemma != "*":
                return str(lemma)
    except Exception:
        pass
    return str(token)


def tokens_to_lemmas(tokens, tagger):
    return [token_to_lemma(token, tagger) for token in tokens]


def evaluate_score_pair_extended(
    scores_o,
    scores_p,
    tokens_o,
    tokens_p,
    lemmas_o,
    lemmas_p,
    k_o,
    k_p,
    gold_sets_o,
):
    scores_o = np.asarray(scores_o, dtype=np.float32)
    scores_p = np.asarray(scores_p, dtype=np.float32)
    lo = min(len(tokens_o), len(lemmas_o), len(scores_o))
    lp = min(len(tokens_p), len(lemmas_p), len(scores_p))
    tokens_o = tokens_o[:lo]
    tokens_p = tokens_p[:lp]
    lemmas_o = lemmas_o[:lo]
    lemmas_p = lemmas_p[:lp]
    scores_o = scores_o[:lo]
    scores_p = scores_p[:lp]

    pred_set = topk_indices(scores_o, k_o)
    f1 = token_f1_wang(pred_set, gold_sets_o)
    map_exact = compute_ranking_map(
        sort_tokens_by_scores(tokens_o, scores_o)[:k_o],
        sort_tokens_by_scores(tokens_p, scores_p)[:k_p],
    )
    map_lemma = compute_ranking_map(
        sort_tokens_by_scores(lemmas_o, scores_o)[:k_o],
        sort_tokens_by_scores(lemmas_p, scores_p)[:k_p],
    )
    return float(f1), float(map_exact), float(map_lemma)


def union_valid_sets(s_all, length):
    out = set()
    for span_set in s_all:
        for index in span_set:
            index = int(index)
            if 0 <= index < length:
                out.add(index)
    return out


def sets_to_binary_scores(s_all, length):
    scores = np.zeros(length, dtype=np.float32)
    members = union_valid_sets(s_all, length)
    if members:
        scores[list(members)] = 1.0
    return scores


def random_contiguous_sets_like(template_sets, length, seed):
    rng = np.random.default_rng(int(seed))
    random_sets = []
    if length <= 0:
        return random_sets
    for template in template_sets:
        span_len = min(max(1, len(template)), length)
        start = int(rng.integers(0, length - span_len + 1))
        random_sets.append(set(range(start, start + span_len)))
    return random_sets


def apply_uniform_set_control(w0, s_all, h=0.5):

    w0 = np.asarray(w0, dtype=np.float32)
    out = w0.copy()
    if len(w0) == 0:
        return out
    members = union_valid_sets(s_all, len(w0))
    if not members:
        return out
    uniform_support = float(np.max(w0))
    for index in members:
        candidate = (1.0 - h) * float(w0[index]) + h * uniform_support
        out[index] = max(float(w0[index]), candidate)
    return out


def calculate_suff_comp_many(words, full_probs, selected_by_setting, predict_batch_fn):




    full_probs = np.asarray(full_probs, dtype=np.float64)
    target = int(np.argmax(full_probs))
    p_full = float(full_probs[target])
    results = {
        name: {"sufficiency": np.nan, "comprehensiveness": np.nan}
        for name in selected_by_setting
    }

    variants = []
    owners = []
    for name, selected in selected_by_setting.items():
        selected = {int(i) for i in selected if 0 <= int(i) < len(words)}
        kept = [token for i, token in enumerate(words) if i in selected]
        removed = [token for i, token in enumerate(words) if i not in selected]
        if not kept or not removed:
            continue
        variants.extend([kept, removed])
        owners.append(name)

    if not variants:
        return results

    variant_probs = np.asarray(predict_batch_fn(variants), dtype=np.float64)
    cursor = 0
    for name in owners:
        keep_probs = variant_probs[cursor]
        remove_probs = variant_probs[cursor + 1]
        cursor += 2
        results[name] = {
            "sufficiency": p_full - float(keep_probs[target]),
            "comprehensiveness": p_full - float(remove_probs[target]),
        }
    return results


def save_extended_results(records):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_df = pd.DataFrame(records)
    result_df.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
    print("rows", len(result_df), "RESULT_CSV", RESULT_CSV)

    added_columns = [
        "f1_extractor_only", "map_exact_extractor_only", "map_lemma_extractor_only",
        "f1_random_span", "map_exact_random_span", "map_lemma_random_span",
        "f1_uniform_set", "map_exact_uniform_set", "map_lemma_uniform_set",
        "f1_oracle", "map_exact_oracle", "map_lemma_oracle",
    ]
    if not result_df.empty:
        for column in added_columns:
            print(column, result_df[column].mean())

def predict_proba_words_batch(model, stoi, sequences):
    input_ids_list = []
    mask_list = []
    unk_id = stoi.get(UNK, 1)
    for words in sequences:
        words = list(words)[:MAX_LEN]
        if not words:
            words = [UNK]
        ids = [stoi.get(token, unk_id) for token in words]
        length = min(len(ids), MAX_LEN)
        ids = ids[:MAX_LEN]
        pad = MAX_LEN - len(ids)
        input_ids_list.append(ids + [0] * pad)
        mask_list.append([True] * length + [False] * pad)

    input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=DEVICE)
    attention_mask = torch.tensor(mask_list, dtype=torch.bool, device=DEVICE)
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        logits = output[0] if isinstance(output, tuple) else output
        return torch.softmax(logits, dim=-1).detach().cpu().numpy()


def main():
    rng = np.random.RandomState(DEBUG_SEED)


    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    stoi = ckpt["stoi"]

    model = BiLSTMAttention(
        vocab_size=len(stoi), emb_dim=128, hidden=196, dropout=0.2, num_labels=2
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()


    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df["tokens_str"] = df["tokens_str"].astype(str)
    df["perturbed_tokens_str"] = df["perturbed_tokens_str"].astype(str)

    validate_extended_columns(df)

    rlr_ratio = compute_dataset_rlr_mean(df)
    print("rlr", rlr_ratio)
    print("MAX_LEN", MAX_LEN)


    spred = load_spred_sets(S_PRED_JSONL)
    print("spred_n", len(spred), "S_PRED_JSONL", S_PRED_JSONL)
    print("H_CONST", H_CONST)
    print("S_DILATE_WINDOW", S_DILATE_WINDOW)


    tagger = Tagger()
    pn_dict = load_pn_table(PN_TABLE_PATH)
    pn_max = max(pn_dict.values()) if pn_dict else 0.0
    print("pn_n", len(pn_dict), "pn_max", pn_max, "PN_ALPHA", PN_ALPHA)


    f1_raw, map_raw = [], []
    f1_pn, map_pn = [], []
    f1_wt0, map_wt0 = [], []
    f1_wt1, map_wt1 = [], []
    records = []

    n_total = 0
    miss_spred_o = 0
    miss_spred_p = 0


    win_delta_max = {w: [] for w in DEBUG_WINDOWS}
    win_topk_change = {w: [] for w in DEBUG_WINDOWS}
    win_map_change = {w: [] for w in DEBUG_WINDOWS}
    debug_printed = 0

    for idx, row in df.iterrows():
        ex_id = int(row["id"])
        tokens_o = str(row["tokens_str"]).split()
        tokens_p = str(row["perturbed_tokens_str"]).split()
        if not tokens_o or not tokens_p:
            continue

        len_o = min(len(tokens_o), MAX_LEN)
        len_p = min(len(tokens_p), MAX_LEN)

        gold_sets_o = build_gold_sets_paired(row, len_o)
        if not gold_sets_o:
            continue

        k_o = max(1, int(round(len_o * rlr_ratio)))
        k_p = max(1, int(round(len_p * rlr_ratio)))


        ids_o = [stoi.get(t, stoi.get(UNK, 1)) for t in tokens_o[:len_o]]
        pad_o = MAX_LEN - len_o
        input_o = torch.tensor([ids_o + [0] * pad_o], device=DEVICE, dtype=torch.long)
        mask_o = torch.tensor([[True] * len_o + [False] * pad_o], device=DEVICE, dtype=torch.bool)


        ids_p = [stoi.get(t, stoi.get(UNK, 1)) for t in tokens_p[:len_p]]
        pad_p = MAX_LEN - len_p
        input_p = torch.tensor([ids_p + [0] * pad_p], device=DEVICE, dtype=torch.long)
        mask_p = torch.tensor([[True] * len_p + [False] * pad_p], device=DEVICE, dtype=torch.bool)


        with torch.no_grad():
            _, w_o = model(input_o, mask_o)
            _, w_p = model(input_p, mask_p)

        w0_o = w_o[0, :len_o].detach().cpu().numpy().astype(np.float32)
        w0_p = w_p[0, :len_p].detach().cpu().numpy().astype(np.float32)


        pred_raw_o = topk_indices(w0_o, k_o)
        f1_raw.append(token_f1_wang(pred_raw_o, gold_sets_o))

        Xo_raw = sort_tokens_by_scores(tokens_o[:len_o], w0_o)[:k_o]
        Xp_raw = sort_tokens_by_scores(tokens_p[:len_p], w0_p)[:k_p]
        map_raw.append(compute_ranking_map(Xo_raw, Xp_raw))


        wpn_o = fuse_att_with_pn(tokens_o[:len_o], w0_o, tagger, pn_dict, pn_max, alpha=PN_ALPHA)
        wpn_p = fuse_att_with_pn(tokens_p[:len_p], w0_p, tagger, pn_dict, pn_max, alpha=PN_ALPHA)

        pred_pn_o = topk_indices(wpn_o, k_o)
        f1_pn.append(token_f1_wang(pred_pn_o, gold_sets_o))

        Xo_pn = sort_tokens_by_scores(tokens_o[:len_o], wpn_o)[:k_o]
        Xp_pn = sort_tokens_by_scores(tokens_p[:len_p], wpn_p)[:k_p]
        map_pn.append(compute_ranking_map(Xo_pn, Xp_pn))


        s_o = spred.get((ex_id, "o"), None)
        s_p = spred.get((ex_id, "p"), None)
        if s_o is None:
            miss_spred_o += 1
            s_o = []
        if s_p is None:
            miss_spred_p += 1
            s_p = []


        wt0_o = apply_weighting(w0_o, s_o, h=H_CONST)
        wt0_p = apply_weighting(w0_p, s_p, h=H_CONST)

        pred_wt0_o = topk_indices(wt0_o, k_o)
        f1_wt0.append(token_f1_wang(pred_wt0_o, gold_sets_o))

        Xo_wt0 = sort_tokens_by_scores(tokens_o[:len_o], wt0_o)[:k_o]
        Xp_wt0 = sort_tokens_by_scores(tokens_p[:len_p], wt0_p)[:k_p]
        map_wt0.append(compute_ranking_map(Xo_wt0, Xp_wt0))


        wt1_o = apply_weighting_ws_dilate(w0_o, s_o, h=H_CONST, window=S_DILATE_WINDOW)
        wt1_p = apply_weighting_ws_dilate(w0_p, s_p, h=H_CONST, window=S_DILATE_WINDOW)

        pred_wt1_o = topk_indices(wt1_o, k_o)
        f1_wt1.append(token_f1_wang(pred_wt1_o, gold_sets_o))

        Xo_wt1 = sort_tokens_by_scores(tokens_o[:len_o], wt1_o)[:k_o]
        Xp_wt1 = sort_tokens_by_scores(tokens_p[:len_p], wt1_p)[:k_p]
        map_wt1.append(compute_ranking_map(Xo_wt1, Xp_wt1))


        if DEBUG:
            wt_base_o = apply_weighting_ws_dilate(w0_o, s_o, h=H_CONST, window=0)
            wt_base_p = apply_weighting_ws_dilate(w0_p, s_p, h=H_CONST, window=0)

            pred_base_o = topk_indices(wt_base_o, k_o)
            Xo_base = sort_tokens_by_scores(tokens_o[:len_o], wt_base_o)[:k_o]
            Xp_base = sort_tokens_by_scores(tokens_p[:len_p], wt_base_p)[:k_p]
            map_base = compute_ranking_map(Xo_base, Xp_base)

            for w in DEBUG_WINDOWS:
                wt_w_o = apply_weighting_ws_dilate(w0_o, s_o, h=H_CONST, window=w)
                wt_w_p = apply_weighting_ws_dilate(w0_p, s_p, h=H_CONST, window=w)

                dmax = float(np.max(np.abs(wt_w_o - wt_base_o)))
                win_delta_max[w].append(dmax)

                pred_w_o = topk_indices(wt_w_o, k_o)
                win_topk_change[w].append(1 if pred_w_o != pred_base_o else 0)

                Xo_w = sort_tokens_by_scores(tokens_o[:len_o], wt_w_o)[:k_o]
                Xp_w = sort_tokens_by_scores(tokens_p[:len_p], wt_w_p)[:k_p]
                map_w = compute_ranking_map(Xo_w, Xp_w)
                win_map_change[w].append(1 if abs(map_w - map_base) > 1e-12 else 0)


        gold_sets_oracle_o = build_gold_sets_paired_strict(
            row, len_o, "aspect_spans", "opinion_spans"
        )
        gold_sets_oracle_p = build_gold_sets_paired_strict(
            row, len_p, "perturbed_aspect_spans", "perturbed_opinion_spans"
        )
        lemmas_o = tokens_to_lemmas(tokens_o[:len_o], tagger)
        lemmas_p = tokens_to_lemmas(tokens_p[:len_p], tagger)

        raw_f1_ex, raw_map_exact, raw_map_lemma = evaluate_score_pair_extended(
            w0_o, w0_p, tokens_o[:len_o], tokens_p[:len_p], lemmas_o, lemmas_p,
            k_o, k_p, gold_sets_o,
        )
        pn_f1_ex, pn_map_exact, pn_map_lemma = evaluate_score_pair_extended(
            wpn_o, wpn_p, tokens_o[:len_o], tokens_p[:len_p], lemmas_o, lemmas_p,
            k_o, k_p, gold_sets_o,
        )
        wt_f1_ex, wt_map_exact, wt_map_lemma = evaluate_score_pair_extended(
            wt0_o, wt0_p, tokens_o[:len_o], tokens_p[:len_p], lemmas_o, lemmas_p,
            k_o, k_p, gold_sets_o,
        )
        wt1_f1_ex, wt1_map_exact, wt1_map_lemma = evaluate_score_pair_extended(
            wt1_o, wt1_p, tokens_o[:len_o], tokens_p[:len_p], lemmas_o, lemmas_p,
            k_o, k_p, gold_sets_o,
        )

        extractor_set_o = union_valid_sets(s_o, len_o)
        extractor_scores_o = sets_to_binary_scores(s_o, len_o)
        extractor_scores_p = sets_to_binary_scores(s_p, len_p)
        _, extractor_map_exact, extractor_map_lemma = evaluate_score_pair_extended(
            extractor_scores_o, extractor_scores_p,
            tokens_o[:len_o], tokens_p[:len_p], lemmas_o, lemmas_p,
            k_o, k_p, gold_sets_o,
        )
        extractor_f1 = token_f1_wang(extractor_set_o, gold_sets_o)

        random_sets_o = random_contiguous_sets_like(
            s_o, len_o, RUN_SEED * 1000003 + ex_id * 2
        )
        random_sets_p = random_contiguous_sets_like(
            s_p, len_p, RUN_SEED * 1000003 + ex_id * 2 + 1
        )
        random_w_o = apply_weighting(w0_o, random_sets_o, h=H_CONST)
        random_w_p = apply_weighting(w0_p, random_sets_p, h=H_CONST)
        random_f1, random_map_exact, random_map_lemma = evaluate_score_pair_extended(
            random_w_o, random_w_p, tokens_o[:len_o], tokens_p[:len_p],
            lemmas_o, lemmas_p, k_o, k_p, gold_sets_o,
        )

        uniform_w_o = apply_uniform_set_control(w0_o, s_o, h=H_CONST)
        uniform_w_p = apply_uniform_set_control(w0_p, s_p, h=H_CONST)
        uniform_f1, uniform_map_exact, uniform_map_lemma = evaluate_score_pair_extended(
            uniform_w_o, uniform_w_p, tokens_o[:len_o], tokens_p[:len_p],
            lemmas_o, lemmas_p, k_o, k_p, gold_sets_o,
        )

        oracle_w_o = apply_weighting(w0_o, gold_sets_oracle_o, h=H_CONST)
        oracle_w_p = apply_weighting(w0_p, gold_sets_oracle_p, h=H_CONST)
        oracle_f1, oracle_map_exact, oracle_map_lemma = evaluate_score_pair_extended(
            oracle_w_o, oracle_w_p, tokens_o[:len_o], tokens_p[:len_p],
            lemmas_o, lemmas_p, k_o, k_p, gold_sets_o,
        )

        predict_batch_fn = lambda sequences: predict_proba_words_batch(model, stoi, sequences)
        full_pair_probs = predict_batch_fn([tokens_o[:len_o], tokens_p[:len_p]])
        pred_o = int(np.argmax(full_pair_probs[0]))
        pred_p = int(np.argmax(full_pair_probs[1]))
        gold_label = int(row["label"])
        correct_o = int(pred_o == gold_label)
        correct_p = int(pred_p == gold_label)
        pair_correct = int(correct_o and correct_p)

        selected_by_setting = {
            "raw": topk_indices(w0_o, k_o),
            "pn": topk_indices(wpn_o, k_o),
            "wt": topk_indices(wt0_o, k_o),
            "ws_dilate": topk_indices(wt1_o, k_o),
            "extractor_only": extractor_set_o,
            "random_span": topk_indices(random_w_o, k_o),
            "uniform_set": topk_indices(uniform_w_o, k_o),
            "oracle": topk_indices(oracle_w_o, k_o),
        }
        suff_comp = calculate_suff_comp_many(
            tokens_o[:len_o], full_pair_probs[0], selected_by_setting, predict_batch_fn
        )

        record = {
            "id": ex_id,
            "seed": RUN_SEED,
            "model": MODEL_NAME,
            "explainer": EXPLAINER_NAME,
            "perturbed_numbers": int(row["perturbed_numbers"]),
            "label": gold_label,
            "pred_o": pred_o,
            "pred_p": pred_p,
            "correct_o": correct_o,
            "correct_p": correct_p,
            "pair_correct": pair_correct,
            "len_o": len_o,
            "len_p": len_p,
            "k_o": k_o,
            "k_p": k_p,
            "f1_raw": raw_f1_ex,
            "map_exact_raw": raw_map_exact,
            "map_lemma_raw": raw_map_lemma,
            "f1_pn": pn_f1_ex,
            "map_exact_pn": pn_map_exact,
            "map_lemma_pn": pn_map_lemma,
            "f1_wt": wt_f1_ex,
            "map_exact_wt": wt_map_exact,
            "map_lemma_wt": wt_map_lemma,
            "f1_ws_dilate": wt1_f1_ex,
            "map_exact_ws_dilate": wt1_map_exact,
            "map_lemma_ws_dilate": wt1_map_lemma,
            "f1_extractor_only": float(extractor_f1),
            "map_exact_extractor_only": extractor_map_exact,
            "map_lemma_extractor_only": extractor_map_lemma,
            "f1_random_span": random_f1,
            "map_exact_random_span": random_map_exact,
            "map_lemma_random_span": random_map_lemma,
            "f1_uniform_set": uniform_f1,
            "map_exact_uniform_set": uniform_map_exact,
            "map_lemma_uniform_set": uniform_map_lemma,
            "f1_oracle": oracle_f1,
            "map_exact_oracle": oracle_map_exact,
            "map_lemma_oracle": oracle_map_lemma,
        }
        for setting_name, values in suff_comp.items():
            record[f"sufficiency_{setting_name}"] = values["sufficiency"]
            record[f"comprehensiveness_{setting_name}"] = values["comprehensiveness"]
        records.append(record)

        n_total += 1


    save_extended_results(records)

    print("raw_f1", np.mean(f1_raw)*100, "raw_map", np.mean(map_raw)*100, "n", len(f1_raw))
    print("pn_f1", np.mean(f1_pn)*100, "pn_map", np.mean(map_pn)*100, "n", len(f1_pn))
    print("wt_f1", np.mean(f1_wt0)*100, "wt_map", np.mean(map_wt0)*100, "n", len(f1_wt0))
    print("ws_f1", np.mean(f1_wt1)*100, "ws_map", np.mean(map_wt1)*100, "n", len(f1_wt1))
    print("miss_o", miss_spred_o, "miss_p", miss_spred_p)


    if DEBUG:
        for w in DEBUG_WINDOWS:
            dm = np.array(win_delta_max[w], dtype=np.float64)
            tc = np.array(win_topk_change[w], dtype=np.float64)
            mc = np.array(win_map_change[w], dtype=np.float64)
            if len(dm) == 0:
                continue
            p95 = float(np.quantile(dm, 0.95))
            print("window", w, "dmax_mean", dm.mean(), "dmax_p95", p95, "topk_change_rate", tc.mean(), "map_change_rate", mc.mean())

if __name__ == "__main__":
    main()