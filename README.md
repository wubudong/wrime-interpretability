# WRIME Interpretability

All scripts use seed 42.

## Functions

- `classifier` trains the three sentiment classifiers.
- `extractor/train_extractor.py` trains the XLM-R and CRF span extractor.
- `span_processing/alignment.py` predicts and aligns spans.
- `span_processing/pairing.py` pairs aspect and opinion spans.
- `classifier` evaluates classifier accuracy.
- `interpretability` runs ATT, IG, and LIME evaluation.

## Data

Perturbed texts are not included because they are derived from WRIME and cannot be redistributed in this repository.

## Order

Run all scripts in this order.

1. `classifier`
2. `extractor/train_extractor.py`
3. `span_processing/alignment.py`
4. `span_processing/pairing.py`
5. `classifier`
6. `interpretability`

## Other Seeds

To use seed 17325 or 48291, replace 42 in each script with the corresponding seed number.
