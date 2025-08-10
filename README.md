# Ner en francais type : CoNLL2003 → spaCy (CamemBERT)

entrainer un model ner francais annotations manuelle par ZIGHED ABDERRAOUF de format  **CoNLL-2003-style** sur des cv francais.
## 1) Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Convert annotations → spaCy
Provide either:
- `data/train.conll` (and optional `data/dev.conll`), **or**
- `data/training_file.json` (JSON with tokens + tags; see accepted formats below).

```bash
# CoNLL -> spaCy
python scripts/conll_to_spacy.py           --train data/train.conll           --dev data/dev.conll           --out-dir data/spacy

# JSON -> spaCy (if you have training_file.json and optionally dev.json)
python scripts/conll_to_spacy.py           --train-json data/training_file.json           --dev-json data/dev.json           --out-dir data/spacy
```

## 3) Train
```bash
bash scripts/train.sh
```

## 4) Evaluate
```bash
python scripts/evaluate.py --model outputs/model-best --dev data/spacy/dev.spacy
```

## 5) Inference (CLI)
```bash
python scripts/infer_cli.py --model outputs/model-best --text "ex : Nadia Martin a 5 ans d'expérience Python chez Thales."
```

## Notes
- Utiliser de  CamemBERT par `spacy-transformers`.
- pour les annotation manuelle demande directement au ZIGHED Abderraouf ( respecte la loi 25 des donnees confidentielle ).
