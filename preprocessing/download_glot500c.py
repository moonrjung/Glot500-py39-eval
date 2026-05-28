"""
Download cis-lmu/Glot500 (public portion) and save each language to disk
in the layout that merge_files.py expects via load_from_disk().
"""
import argparse
import os
import pandas as pd
from datasets import load_dataset, get_dataset_config_names
from huggingface_hub import HfApi


#MJ: python download_glot500c.py --max_langs 5

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",       default="/disk3/moon/Glot500/miscellaneous/languages_stats.csv")
    ap.add_argument("--out_dir",   default="/disk3/moon/Glot500/data/raw")
    ap.add_argument("--cache_dir", default="/disk3/moon/Glot500/cache/hf_datasets")
    ap.add_argument("--max_langs", type=int, default=None,
                    help="Limit how many languages to download (for testing).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # 1. Build the list of languages merge_files.py will look for:
    #    seen (no count filter) + unseen with count >= 30000
    df = pd.read_csv(args.csv)
    def key(lg, script):
        return f"{lg}_{script.replace('[', '').replace(']', '').replace(chr(39), '')}"

    wanted = set()
    for lg, script, count, is_seen in zip(
            df["language"], df["script"], df["new_length"], df["XLM-R"]):
        if is_seen is True or count >= 30000:
            wanted.add(key(lg, script))
    print(f"Total languages to fetch: {len(wanted)}") #MJ: Total languages to fetch: 570

    #MJ: 
    # Result: 100 languages have XLM-R == True
    # Three ways to compute it, all give the same answer:

    # Method 1 — Vectorized boolean (one-liner, fastest)

    # n_seen = (df['XLM-R'] == True).sum()
    # # 100

   # n = ((df['XLM-R'] == True) & (df['new_length'] < 30000)).sum()
   # 8
    # language script       new_length
    # ----------------------------------
    # ori      ['Latn']          9,166
    # hau      ['Arab']          9,593
    # uig      ['Latn']          9,637
    # som      ['Arab']         14,199
    # lao      ['Latn']         19,121
    # kir      ['Latn']         20,440
    # san      ['Latn']         25,742
    # prs      ['Arab']         26,823



    # 2. Find which of those configs actually exist on the public hub.
    # api = HfApi()
    # info = api.dataset_info("cis-lmu/Glot500")

    # available = {c.split("/")[-1] if "/" in c else c for c in info.config_names}


    available = set(get_dataset_config_names("cis-lmu/Glot500"))

    to_fetch = sorted(wanted & available)
    missing  = sorted(wanted - available)
    print(f"Available on public hub:  {len(to_fetch)}")
    print(f"Missing (restricted):     {len(missing)}  e.g. {missing[:8]}")

    if args.max_langs:
        to_fetch = to_fetch[:args.max_langs]
        print(f"Limiting to first {args.max_langs} languages")

    # 3. Pull each language config, save to disk in load_from_disk() format.
    for i, lg in enumerate(to_fetch, 1):
        out_path = os.path.join(args.out_dir, lg)
        if os.path.exists(os.path.join(out_path, "dataset_info.json")):
            print(f"[{i}/{len(to_fetch)}] {lg:<15} already on disk, skipping")
            continue
        try:

            ds = load_dataset("cis-lmu/Glot500", lg, cache_dir=args.cache_dir)

            ds.save_to_disk(out_path)
            n = len(ds["train"]) if "train" in ds else 0
            print(f"[{i}/{len(to_fetch)}] {lg:<15} saved ({n:,} rows)")
        except Exception as e:
            print(f"[{i}/{len(to_fetch)}] {lg:<15} FAILED: {e}")


if __name__ == "__main__":
    main()

# 1) When load_dataset() is the right tool
# Use it whenever the dataset is set up the conventional way:

# The dataset declares named configs in its loading script or YAML frontmatter (wikiann, glue, wikitext, xnli, ...).
# Files are referenced by a Python loader script ({repo}/{name}.py) or follow a standard layout (Parquet auto-discovery).
# You want a fully-typed, ready-to-iterate Dataset object — not raw files.
# Example (the textbook case):


# from datasets import load_dataset

# ds = load_dataset("wikiann", "en")           # 'en' is a real config name
# print(ds["train"][0])                         # immediately usable
# # {'tokens': ['Barack', 'Obama', ...], 'ner_tags': [1, 2, ...], ...}
# load_dataset does everything for you: download → parse → type-cast → cache → return a typed object.

# 2) When snapshot_download() is the right tool
# Use it when:

# The dataset is essentially a file dump with a directory structure but no formal config declarations — like cis-lmu/Glot500, where each language is a subdirectory of pre-built Arrow files.
# You want to download only a subset of files (allow_patterns="abk_Cyrl/*") instead of the whole repo.
# The dataset's loader script is broken, missing, or incompatible with your datasets version.
# You need files in a specific local layout that you'll then process yourself — e.g., feed into load_from_disk(), your own custom code, or a non-HF tool.
# Example (what we need for Glot500):


# from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="cis-lmu/Glot500",
#     repo_type="dataset",
#     allow_patterns="abk_Cyrl/*",       # just this language
#     local_dir="/disk3/moon/Glot500/data/raw",
#     local_dir_use_symlinks=False,
# )
# # Now files are at /disk3/moon/Glot500/data/raw/abk_Cyrl/train/...
# # Use them however you want — load_from_disk(), tar them up, etc.
# It just mirrors files. Nothing parsed, nothing typed. You're in charge of what happens next.

# 3) How to decide
# Three quick checks for any dataset you encounter:

# Does get_dataset_config_names("repo_id") return real names?

# Returns ['en', 'fr', ...] or ['cola', 'sst2', ...] → use load_dataset with one of those names.
# Returns [repo_id] (just a placeholder) or [] → the dataset doesn't expose configs. Use snapshot_download to get the files directly.
# Does the dataset page on HF Hub have a working "Viewer"?

# Yes → load_dataset will probably work.
# No, viewer says "this dataset can't be auto-detected" → you'll likely need snapshot_download plus manual loading.
# Are the files already in a HF-native format (.arrow, dataset_info.json)?

# Yes → snapshot_download then load_from_disk(local_path) is the cleanest path. You get the dataset's typed structure back without load_dataset's parsing logic in the middle.
# No (raw .txt, .csv, .jsonl) → you'll need either load_dataset (if configs work) or snapshot_download followed by your own Dataset.from_csv(...) / Dataset.from_json(...).


# 4)For cis-lmu/Glot500 specifically, snapshot_download is correct because the dataset is a directory tree of pre-built .arrow files with no config declarations. load_dataset can't see the per-language structure and falls into the "load everything as one dataset" failure mode you observed.
