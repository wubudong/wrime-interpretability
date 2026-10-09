from pathlib import Path

import os, json

PROJECT_ROOT = Path(os.environ.get("WRIME_EXP_ROOT", ".")).resolve()

IN_JSONL  = str(PROJECT_ROOT / "outputs" / "spred_cache" / "spred_tcsv_aligned_seed42.jsonl")
OUT_JSONL = str(PROJECT_ROOT / "outputs" / "spred_cache" / "spred_tcsv_paired_mark_unpaired_seed42.jsonl")

def spans_from_tagseq(tags, typ):
    spans = []
    cur = None
    s = None
    for i, tg in enumerate(tags + ["O"]):
        if tg == "O" or tg is None:
            if cur == typ:
                spans.append((s, i - 1))
            cur, s = None, None
            continue

        if "-" not in tg:
            if cur == typ:
                spans.append((s, i - 1))
            cur, s = None, None
            continue

        pref, t = tg.split("-", 1)
        if t != typ:
            if cur == typ:
                spans.append((s, i - 1))
            cur, s = None, None
            continue

        if pref == "B" or cur != typ:
            if cur == typ:
                spans.append((s, i - 1))
            cur, s = typ, i
        elif pref == "I":
            cur = typ

    return spans

def to_set_from_span(span):
    a_s, a_e = span
    return set(range(a_s, a_e + 1))

def main():
    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)

    n = 0
    with open(IN_JSONL, "r", encoding="utf-8") as fin, open(OUT_JSONL, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            tags = obj.get("pred_tags", None)
            if tags is None:
                raise ValueError("error!error!")

            aspects  = spans_from_tagseq(tags, "Aspect")
            opinions = spans_from_tagseq(tags, "Opinion")
            aspects.sort(key=lambda x: x[0])
            opinions.sort(key=lambda x: x[0])

            paired_sets = []
            unpaired_sets = []

            m = min(len(aspects), len(opinions))

            for i in range(m):
                sset = to_set_from_span(aspects[i]) | to_set_from_span(opinions[i])
                paired_sets.append(sorted(sset))


            for i in range(m, len(aspects)):
                unpaired_sets.append(sorted(to_set_from_span(aspects[i])))
            for i in range(m, len(opinions)):
                unpaired_sets.append(sorted(to_set_from_span(opinions[i])))

            obj["s_pred_sets"] = paired_sets + unpaired_sets
            obj["s_pred_unpaired_sets"] = unpaired_sets
            obj["pairing_rule"] = "order_zip_pair_min_keep_rest_as_unpaired"

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1

    print("n", n, "OUT_JSONL", OUT_JSONL)

if __name__ == "__main__":
    main()