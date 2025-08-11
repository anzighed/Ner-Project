import argparse, json
import spacy

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    args = ap.parse_args()

    nlp = spacy.load(args.model)
    doc = nlp(args.text)
    ents = [{"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char} for e in doc.ents]
    print(json.dumps({"text": doc.text, "entities": ents}, ensure_ascii=False, indent=2))
