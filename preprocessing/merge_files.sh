#!/bin/bash
# Must be run from /disk3/moon/Glot500/preprocessing/
# (merge_files.py reads ../miscellaneous/languages_stats.csv via a relative path)
#
# Both --data_directory and --save_directory MUST end with a trailing slash
# (merge_files.py uses naked string concatenation: save_directory + 'Glot500.txt').
#
# --data_directory expects per-language HuggingFace datasets directories
# (Arrow format), one per language, named {lang_code}_{script} (e.g. eng_Latn).
# Each must have a 'train' split with a 'text' column.
#
# Output: /disk3/moon/Glot500/data/Glot500.txt

python merge_files.py \
  --data_directory     /disk3/moon/Glot500/data/raw/ \
  --save_directory     /disk3/moon/Glot500/data/ \
  --experiment_name    Glot500 \
  --lg_sampling_factor 0.3 \
  --scale              30


# Where the data lives
# Resource	URL	Coverage	How to get it
# Glot500-C (public)	cis-lmu/Glot500 on HF Hub	Redistributable portion only (~75% of languages, mostly non-religious / open-license sources)	datasets.load_dataset(...) — no login needed
# Glot500-C (full)	Sent by the cisnlp authors after you fill in the data request form	All languages including the Parallel Bible Corpus and other restricted sources	Email-based — usually answered in a day or so
# GlotCC-V1 (CommonCrawl-derived)	cis-lmu/GlotCC-V1	Additional CommonCrawl text per language, language-identified	Public, larger than Glot500-C
# GlotLID-Corpus	cis-lmu/glotlid-corpus	Gated. Useful for LID training, not directly for Glot500-m pretraining.	Fill the gated-access form (1-day approval)
