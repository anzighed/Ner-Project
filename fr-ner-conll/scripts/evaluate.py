import argparse
import spacy
from spacy.tokens import DocBin

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dev", required=True)
    args = ap.parse_args()

    nlp = spacy.load(args.model)
    db = DocBin().from_disk(args.dev)
    docs = list(db.get_docs(nlp.vocab))

    tp = fp = fn = 0
    for gold in docs:
        pred = nlp(gold.text)
        gold_ents = {(e.start_char, e.end_char, e.label_) for e in gold.ents}
        pred_ents = {(e.start_char, e.end_char, e.label_) for e in pred.ents}
        tp += len(gold_ents & pred_ents)
        fp += len(pred_ents - gold_ents)
        fn += len(gold_ents - pred_ents)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    print({"precision": precision, "recall": recall, "f1": f1})
