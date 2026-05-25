#!/usr/bin/env python3
"""Fetch the PAN-X / WikiAnn NER data from HuggingFace and write per-language
files in the legacy XTREME layout that `panx_preprocess` in
`utils_preprocess.py` consumes.

For every language code in --lang_list this writes three files into
--output_dir:

    {lg}-train
    {lg}-dev      (HF "validation" split)
    {lg}-test

Each file is in the original PAN-X CoNLL format:

    {lg}:{token}\\t{BIO-tag}

with an empty line between sentences. Failures on individual languages are
logged but do not stop the run, because a handful of the language codes used
by Glot500 are not present in the HuggingFace `wikiann` configuration set.
"""
import argparse
import os
import sys

from datasets import load_dataset


TAG_NAMES = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
HF_SPLIT = {"train": "train", "dev": "validation", "test": "test"}


def write_split(ds, out_path, lang):
    with open(out_path, "w", encoding="utf-8") as fout:
        for ex in ds:
            for tok, tag_id in zip(ex["tokens"], ex["ner_tags"]):
                tag = TAG_NAMES[tag_id] if isinstance(tag_id, int) else tag_id
                fout.write(f"{lang}:{tok}\t{tag}\n")
            fout.write("\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang_list", required=True,
                   help="Text file with one language code per line.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--hf_name", default="wikiann",
                   help="HuggingFace dataset name (default: wikiann).")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.lang_list) as f:
        langs = [l.strip() for l in f if l.strip()]

    failures = []
    for lg in langs:
        try:
            dsets = load_dataset(args.hf_name, lg)
        except Exception as e:
            print(f"[WARN] failed to load {lg}: {e}", file=sys.stderr)
            failures.append(lg)
            continue
        for our_split, hf_split in HF_SPLIT.items():
            if hf_split not in dsets:
                print(f"[WARN] {lg} has no split {hf_split}", file=sys.stderr)
                continue
            write_split(dsets[hf_split],
                        os.path.join(args.output_dir, f"{lg}-{our_split}"),
                        lg)
        print(f"[OK] {lg}")

    if failures:
        print(f"\n{len(failures)} languages failed (continuing anyway): {failures}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
