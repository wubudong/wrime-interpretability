from pathlib import Path















import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import re
import json
import time

import numpy as np
import pandas as pd
import torch
from fugashi import Tagger
from transformers import AutoTokenizer, AutoModelForSequenceClassification





PROJECT_ROOT = Path(os.environ.get("WRIME_EXP_ROOT", ".")).resolve()

MODEL_DIR = str(PROJECT_ROOT / "outputs" / "baseline" / "base_seed42")


TOKENIZER_ID = "rinna/japanese-roberta-base"

CSV_PATH = str(PROJECT_ROOT / "data" / "processed" / "t_eval_manual" / "t.csv")
S_PRED_JSONL = str(PROJECT_ROOT / "outputs" / "spred_cache" / "spred_tcsv_paired_mark_unpaired_seed42.jsonl")

PN_TABLE_PATH = str(PROJECT_ROOT / "data" / "raw3" / "pn_ja.dic.txt")
PN_ALPHA = 0.5




H_CONST = 0.5
RUN_SEED = 42
MODEL_NAME = "roberta_base"
EXPLAINER_NAME = "att"
RESULTS_DIR = str(PROJECT_ROOT / "outputs" / "table4_per_instance" / f"seed{RUN_SEED}")
RESULT_CSV = os.path.join(
    RESULTS_DIR, f"{MODEL_NAME}_{EXPLAINER_NAME}_seed{RUN_SEED}_per_instance.csv"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_WORD_LEN = 128
MAX_SUBWORD_LEN = 512

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False




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

def parse_span_str_to_slots(span_str: str):
    if span_str is None or (isinstance(span_str, float) and pd.isna(span_str)):
        return [set()]
    s = str(span_str).strip()
    if not s or s.lower() == "(none)":
        return [set()]
    return [parse_one_span_part(p) for p in s.split("|")]

def build_gold_sets_paired(row, actual_len: int):
    a_slots = parse_span_str_to_slots(row.get("aspect_spans", ""))
    o_slots = parse_span_str_to_slots(row.get("opinion_spans", ""))
    gold_sets = []
    for i in range(max(len(a_slots), len(o_slots))):
        a_set = a_slots[i] if i < len(a_slots) else set()
        o_set = o_slots[i] if i < len(o_slots) else set()
        s = {t for t in (a_set | o_set) if 0 <= t < actual_len}
        if s:
            gold_sets.append(s)
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
        inter = len(pred_set & set(g))
        if inter == 0:
            continue
        p = inter / len(pred_set)
        r = inter / len(g)
        best = max(best, 2 * p * r / (p + r))
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
    L = min(len(tokens), len(scores))
    tokens = tokens[:L]
    scores = scores[:L]
    order = np.argsort(-scores, kind="stable")
    return [tokens[i] for i in order]

def compute_dataset_rlr_mean(df):
    ratios = []
    for _, row in df.iterrows():
        toks = str(row["tokens_str"]).split()
        if not toks:
            continue
        L = min(len(toks), MAX_WORD_LEN)
        gold_sets = build_gold_sets_paired(row, L)
        if not gold_sets:
            continue
        ratios.append(len(union_rationales(gold_sets)) / L)
    return float(np.mean(ratios)) if ratios else 0.0




def load_spred_sets(path: str):
    mp = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            mp[(int(o["id"]), o["which"])] = [set(map(int, s)) for s in o.get("s_pred_sets", [])]
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
                W_lists[t].append((1 - h) * float(w0[t]) + h * ws)
    return np.array([max(x) for x in W_lists], dtype=np.float32)

def apply_weighting_ws_dilate_w1(w0: np.ndarray, s_all, h=0.5):
    w0 = np.asarray(w0, dtype=np.float32)
    L = len(w0)
    W_lists = [[float(v)] for v in w0.tolist()]
    for s in s_all:
        dil = set()
        for i in s:
            for d in (-1, 0, 1):
                j = i + d
                if 0 <= j < L:
                    dil.add(j)
        ws = float(np.sum(w0[list(dil)])) if dil else 0.0
        for t in s:
            if 0 <= t < L:
                W_lists[t].append((1 - h) * float(w0[t]) + h * ws)
    return np.array([max(x) for x in W_lists], dtype=np.float32)




def load_pn_table(path):
    pn = {}
    for enc in ["utf-8", "shift_jis", "cp932"]:
        try:
            with open(path, "r", encoding=enc) as f:
                for line in f:
                    ps = line.strip().split(":")
                    if len(ps) >= 4:
                        pn[ps[0]] = abs(float(ps[3]))
            return pn
        except Exception:
            continue
    return {}

def fuse_with_pn(tokens, w0, tagger, pn_dict, pn_max, alpha=0.5):
    w0 = np.asarray(w0, dtype=np.float32)
    m = float(np.max(w0)) + 1e-9
    w0n = w0 / m
    pn_scores = np.zeros_like(w0n, dtype=np.float32)
    for i, t in enumerate(tokens):
        try:
            node = tagger(t)[0]
            lemma = node.feature.lemma if getattr(node.feature, "lemma", None) else t
        except Exception:
            lemma = t
        pn_scores[i] = float(pn_dict.get(lemma, pn_dict.get(t, 0.0)))
    if pn_max > 0:
        pn_scores = pn_scores / float(pn_max)
    return ((1 - alpha) * w0n + alpha * pn_scores).astype(np.float32)




def att_word_scores_roberta(model, tokenizer, words):






    words = words[:MAX_WORD_LEN]
    max_pos = int(getattr(model.config, "max_position_embeddings", MAX_SUBWORD_LEN))
    max_len = min(MAX_SUBWORD_LEN, max_pos)

    enc = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_len,
        padding=False,
        return_tensors="pt",
        return_attention_mask=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("error!error!")

    word_ids = enc.word_ids(batch_index=0)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc, output_attentions=True, return_dict=True)
        attns = out.attentions

    last = attns[-1][0]
    cls_to_all = last[:, 0, :]
    scores_sub = cls_to_all.mean(0).detach().cpu().numpy().astype(np.float32)

    word_scores = np.zeros(MAX_WORD_LEN, dtype=np.float32)
    max_wi = -1
    for si, wi in enumerate(word_ids):
        if wi is None:
            continue
        if wi >= MAX_WORD_LEN:
            continue
        max_wi = max(max_wi, wi)
        word_scores[wi] += scores_sub[si]

    Lw = max_wi + 1 if max_wi >= 0 else 0
    return word_scores[:Lw]








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

def predict_proba_words_batch(model, tokenizer, sequences):
    batch_words = []
    fallback = getattr(tokenizer, "unk_token", None) or "<unk>"
    for words in sequences:
        words = list(words)[:MAX_WORD_LEN]
        batch_words.append(words if words else [fallback])

    max_pos = int(getattr(model.config, "max_position_embeddings", MAX_SUBWORD_LEN))
    max_len = min(MAX_SUBWORD_LEN, max_pos)
    enc = tokenizer(
        batch_words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_len,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    enc = {key: value.to(DEVICE) for key, value in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        return torch.softmax(logits, dim=-1).detach().cpu().numpy()


def main():
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError("error!error!")

    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df["tokens_str"] = df["tokens_str"].astype(str)
    df["perturbed_tokens_str"] = df["perturbed_tokens_str"].astype(str)

    validate_extended_columns(df)
    rlr = compute_dataset_rlr_mean(df)
    print("rlr", rlr, "DEVICE", DEVICE)

    spred = load_spred_sets(S_PRED_JSONL)
    print("spred_n", len(spred), "S_PRED_JSONL", S_PRED_JSONL)

    tagger = Tagger()
    pn_dict = load_pn_table(PN_TABLE_PATH)
    pn_max = max(pn_dict.values()) if pn_dict else 0.0
    print("pn_n", len(pn_dict), "pn_max", pn_max, "PN_ALPHA", PN_ALPHA)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("error!error!")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()

    print("MODEL_DIR", MODEL_DIR, "DEVICE", DEVICE)
    print("TOKENIZER_ID", TOKENIZER_ID, "is_fast", getattr(tokenizer, "is_fast", False))

    f1_raw, map_raw = [], []
    f1_pn, map_pn = [], []
    f1_wt, map_wt = [], []
    f1_wt1, map_wt1 = [], []
    records = []

    miss_o = miss_p = 0
    n = 0
    t0 = time.time()

    for _, row in df.iterrows():
        ex_id = int(row["id"])
        toks_o = str(row["tokens_str"]).split()
        toks_p = str(row["perturbed_tokens_str"]).split()
        if not toks_o or not toks_p:
            continue

        len_o = min(len(toks_o), MAX_WORD_LEN)
        len_p = min(len(toks_p), MAX_WORD_LEN)
        toks_o = toks_o[:len_o]
        toks_p = toks_p[:len_p]

        gold_sets_o = build_gold_sets_paired(row, len_o)
        if not gold_sets_o:
            continue

        w0_o = att_word_scores_roberta(model, tokenizer, toks_o)
        w0_p = att_word_scores_roberta(model, tokenizer, toks_p)
        if len(w0_o) == 0 or len(w0_p) == 0:
            continue

        if len(w0_o) < len_o:
            toks_o = toks_o[:len(w0_o)]
            len_o = len(toks_o)
            gold_sets_o = build_gold_sets_paired(row, len_o)
            if not gold_sets_o:
                continue
        if len(w0_p) < len_p:
            toks_p = toks_p[:len(w0_p)]
            len_p = len(toks_p)

        k_o = max(1, int(round(len_o * rlr)))
        k_p = max(1, int(round(len_p * rlr)))


        f1_raw.append(token_f1_wang(topk_indices(w0_o, k_o), gold_sets_o))
        map_raw.append(
            compute_ranking_map(
                sort_tokens_by_scores(toks_o, w0_o)[:k_o],
                sort_tokens_by_scores(toks_p, w0_p)[:k_p],
            )
        )


        wpn_o = fuse_with_pn(toks_o, w0_o, tagger, pn_dict, pn_max, alpha=PN_ALPHA)
        wpn_p = fuse_with_pn(toks_p, w0_p, tagger, pn_dict, pn_max, alpha=PN_ALPHA)
        f1_pn.append(token_f1_wang(topk_indices(wpn_o, k_o), gold_sets_o))
        map_pn.append(
            compute_ranking_map(
                sort_tokens_by_scores(toks_o, wpn_o)[:k_o],
                sort_tokens_by_scores(toks_p, wpn_p)[:k_p],
            )
        )


        s_o = spred.get((ex_id, "o"), None)
        s_p = spred.get((ex_id, "p"), None)
        if s_o is None:
            miss_o += 1
            s_o = []
        if s_p is None:
            miss_p += 1
            s_p = []


        wt_o = apply_weighting(w0_o, s_o, h=H_CONST)
        wt_p = apply_weighting(w0_p, s_p, h=H_CONST)
        f1_wt.append(token_f1_wang(topk_indices(wt_o, k_o), gold_sets_o))
        map_wt.append(
            compute_ranking_map(
                sort_tokens_by_scores(toks_o, wt_o)[:k_o],
                sort_tokens_by_scores(toks_p, wt_p)[:k_p],
            )
        )


        wt1_o = apply_weighting_ws_dilate_w1(w0_o, s_o, h=H_CONST)
        wt1_p = apply_weighting_ws_dilate_w1(w0_p, s_p, h=H_CONST)
        f1_wt1.append(token_f1_wang(topk_indices(wt1_o, k_o), gold_sets_o))
        map_wt1.append(
            compute_ranking_map(
                sort_tokens_by_scores(toks_o, wt1_o)[:k_o],
                sort_tokens_by_scores(toks_p, wt1_p)[:k_p],
            )
        )



        gold_sets_oracle_o = build_gold_sets_paired_strict(
            row, len_o, "aspect_spans", "opinion_spans"
        )
        gold_sets_oracle_p = build_gold_sets_paired_strict(
            row, len_p, "perturbed_aspect_spans", "perturbed_opinion_spans"
        )
        lemmas_o = tokens_to_lemmas(toks_o[:len_o], tagger)
        lemmas_p = tokens_to_lemmas(toks_p[:len_p], tagger)

        raw_f1_ex, raw_map_exact, raw_map_lemma = evaluate_score_pair_extended(
            w0_o, w0_p, toks_o[:len_o], toks_p[:len_p], lemmas_o, lemmas_p,
            k_o, k_p, gold_sets_o,
        )
        pn_f1_ex, pn_map_exact, pn_map_lemma = evaluate_score_pair_extended(
            wpn_o, wpn_p, toks_o[:len_o], toks_p[:len_p], lemmas_o, lemmas_p,
            k_o, k_p, gold_sets_o,
        )
        wt_f1_ex, wt_map_exact, wt_map_lemma = evaluate_score_pair_extended(
            wt_o, wt_p, toks_o[:len_o], toks_p[:len_p], lemmas_o, lemmas_p,
            k_o, k_p, gold_sets_o,
        )
        wt1_f1_ex, wt1_map_exact, wt1_map_lemma = evaluate_score_pair_extended(
            wt1_o, wt1_p, toks_o[:len_o], toks_p[:len_p], lemmas_o, lemmas_p,
            k_o, k_p, gold_sets_o,
        )

        extractor_set_o = union_valid_sets(s_o, len_o)
        extractor_scores_o = sets_to_binary_scores(s_o, len_o)
        extractor_scores_p = sets_to_binary_scores(s_p, len_p)
        _, extractor_map_exact, extractor_map_lemma = evaluate_score_pair_extended(
            extractor_scores_o, extractor_scores_p,
            toks_o[:len_o], toks_p[:len_p], lemmas_o, lemmas_p,
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
            random_w_o, random_w_p, toks_o[:len_o], toks_p[:len_p],
            lemmas_o, lemmas_p, k_o, k_p, gold_sets_o,
        )

        uniform_w_o = apply_uniform_set_control(w0_o, s_o, h=H_CONST)
        uniform_w_p = apply_uniform_set_control(w0_p, s_p, h=H_CONST)
        uniform_f1, uniform_map_exact, uniform_map_lemma = evaluate_score_pair_extended(
            uniform_w_o, uniform_w_p, toks_o[:len_o], toks_p[:len_p],
            lemmas_o, lemmas_p, k_o, k_p, gold_sets_o,
        )

        oracle_w_o = apply_weighting(w0_o, gold_sets_oracle_o, h=H_CONST)
        oracle_w_p = apply_weighting(w0_p, gold_sets_oracle_p, h=H_CONST)
        oracle_f1, oracle_map_exact, oracle_map_lemma = evaluate_score_pair_extended(
            oracle_w_o, oracle_w_p, toks_o[:len_o], toks_p[:len_p],
            lemmas_o, lemmas_p, k_o, k_p, gold_sets_o,
        )

        predict_batch_fn = lambda sequences: predict_proba_words_batch(model, tokenizer, sequences)
        full_pair_probs = predict_batch_fn([toks_o[:len_o], toks_p[:len_p]])
        pred_o = int(np.argmax(full_pair_probs[0]))
        pred_p = int(np.argmax(full_pair_probs[1]))
        gold_label = int(row["label"])
        correct_o = int(pred_o == gold_label)
        correct_p = int(pred_p == gold_label)
        pair_correct = int(correct_o and correct_p)

        selected_by_setting = {
            "raw": topk_indices(w0_o, k_o),
            "pn": topk_indices(wpn_o, k_o),
            "wt": topk_indices(wt_o, k_o),
            "ws_dilate": topk_indices(wt1_o, k_o),
            "extractor_only": extractor_set_o,
            "random_span": topk_indices(random_w_o, k_o),
            "uniform_set": topk_indices(uniform_w_o, k_o),
            "oracle": topk_indices(oracle_w_o, k_o),
        }
        suff_comp = calculate_suff_comp_many(
            toks_o[:len_o], full_pair_probs[0], selected_by_setting, predict_batch_fn
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

        n += 1
    save_extended_results(records)

    print("raw_f1", np.mean(f1_raw)*100, "raw_map", np.mean(map_raw)*100, "n", len(f1_raw))
    print("pn_f1", np.mean(f1_pn)*100, "pn_map", np.mean(map_pn)*100, "n", len(f1_pn))
    print("wt_f1", np.mean(f1_wt)*100, "wt_map", np.mean(map_wt)*100, "n", len(f1_wt))
    print("ws_f1", np.mean(f1_wt1)*100, "ws_map", np.mean(map_wt1)*100, "n", len(f1_wt1))
    print("miss_o", miss_o, "miss_p", miss_p)

if __name__ == "__main__":
    main()