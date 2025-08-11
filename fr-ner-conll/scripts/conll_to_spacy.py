import argparse
from pathlib import Path
import json
import spacy
from spacy.tokens import Doc, DocBin

SEP = "\n\n"

def read_conll(path: Path):
    sents = []
    cur = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            if cur:
                sents.append(cur); cur = []
            continue
        parts = line.split()
        token = parts[0]
        tag = parts[-1]
        cur.append((token, tag))
    if cur:
        sents.append(cur)
    return sents

def read_json(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    sents = []
    if isinstance(data, list) and data and isinstance(data[0], dict):
        for item in data:
            toks = item.get("tokens") or []
            tags = item.get("tags") or []
            sents.append(list(zip(toks, tags)))
    elif isinstance(data, list) and data and isinstance(data[0], list):
        for sent in data:
            sents.append([(tok, tag) for tok, tag in sent])
    else:
        raise ValueError("Unsupported JSON structure")
    return sents

def sents_to_docbin(nlp, sents):
    db = DocBin(store_user_data=False)
    labels = set()
    for sent in sents:
        tokens = [t for t,_ in sent]
        tags = [y for _,y in sent]
        doc = Doc(nlp.vocab, words=tokens)
        ents_idx = []
        start = None; ent_label = None
        for i, tag in enumerate(tags + ["O"]):  # sentinel
            if tag.startswith("B-"):
                if start is not None:
                    ents_idx.append((start, i, ent_label))
                start, ent_label = i, tag[2:]
                labels.add(ent_label)
            elif tag.startswith("I-"):
                pass
            else:
                if start is not None:
                    ents_idx.append((start, i, ent_label))
                    start, ent_label = None, None
        spans = []
        for i,j,label in ents_idx:
            span = doc[i:j].as_span(label=label)
            if span is not None:
                spans.append(span)
        doc.ents = spans
        db.add(doc)
    return db, sorted(labels)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path)
    ap.add_argument("--dev", type=Path)
    ap.add_argument("--train-json", type=Path)
    ap.add_argument("--dev-json", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--split", type=float, default=0.9)
    args = ap.parse_args()

    nlp = spacy.blank("fr")

    if args.train and args.train.exists():
        s_train = read_conll(args.train)
        s_dev = read_conll(args.dev) if (args.dev and args.dev.exists()) else []
    elif args.train_json and args.train_json.exists():
        s_train = read_json(args.train_json)
        s_dev = read_json(args.dev_json) if (args.dev_json and args.dev_json.exists()) else []
    else:
        raise SystemExit("Provide --train/--dev (CoNLL) or --train-json/--dev-json (JSON)")

    if not s_dev:
        cut = int(len(s_train) * args.split)
        s_train, s_dev = s_train[:cut], s_train[cut:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    db_train, labels = sents_to_docbin(nlp, s_train)
    db_dev, _ = sents_to_docbin(nlp, s_dev)

    db_train.to_disk(args.out_dir / "train.spacy")
    db_dev.to_disk(args.out_dir / "dev.spacy")
    (args.out_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote train/dev .spacy. Labels: {labels}")
