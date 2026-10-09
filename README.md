# WRIME Interpretability

All scripts use seed 42.

## Functions

- `classifier/train_bilstm.py`, `classifier/train_roberta_base.py`, `classifier/train_roberta_large.py` train the three sentiment classifiers.
- `extractor/train_extractor.py` trains the XLM-R and CRF span extractor.
- `span_processing/alignment.py` predicts and aligns spans.
- `span_processing/pairing.py` pairs aspect and opinion spans.
- `classifier/eval_bilstm.py`, `classifier/eval_roberta_base.py`, `classifier/eval_roberta_large.py` evaluate classifier accuracy.
- `interpretability` runs ATT, IG, and LIME evaluation.

## Data

Perturbed texts are not included because they are derived from WRIME and cannot be redistributed in this repository.

## Order

Run all scripts in this order.

1. `classifier/train_bilstm.py`, `classifier/train_roberta_base.py`, `classifier/train_roberta_large.py`
2. `extractor/train_extractor.py`
3. `span_processing/alignment.py`
4. `span_processing/pairing.py`
5. `classifier/eval_bilstm.py`, `classifier/eval_roberta_base.py`, `classifier/eval_roberta_large.py`
6. `interpretability`

## Other Seeds

To use seed 17325 or 48291, replace 42 in each script with the corresponding seed number.
