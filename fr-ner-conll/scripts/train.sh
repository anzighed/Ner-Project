#!/usr/bin/env bash
set -euo pipefail
CONFIG=configs/config_camembert.cfg
OUT=outputs
python -m spacy train "$CONFIG" --output "$OUT"           --paths.train data/spacy/train.spacy           --paths.dev data/spacy/dev.spacy
