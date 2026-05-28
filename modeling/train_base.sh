#!/usr/bin/env bash
# Glot500 base-size pretraining on 1 × RTX A6000 (48 GB), pinned to GPU 1
# Effective batch size: 12 (per-device) × 32 (accum) × 1 (GPU) = 384 (matches upstream)

CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true python run.py \
  --model_name_or_path xlm-roberta-base \
  --train_file       /home/sogang/mnt/db_1/moon/Glot500/data/Glot500.txt \
  --tokenizer_name   /home/sogang/mnt/db_1/moon/Glot500/tokenization/output/Glot500_extended_spm \
  --output_dir       /home/sogang/mnt/db_1/moon/Glot500/runs/glot500-base \
  --cache_dir        /home/sogang/mnt/db_1/moon/Glot500/cache \
  --per_device_train_batch_size 12 \
  --gradient_accumulation_steps 32 \
  --fp16 True \
  --do_train \
  --num_train_epochs 100 \
  --save_steps 10000


#MJ:  What you still need to build before this can run
# 1. Train the Glot500 tokenizer (tokenization/train.sh)
# This produces the expanded-vocab SentencePiece model + tokenizer files that --tokenizer_name points at. Look inside tokenization/train.sh — it'll call tokenization/run.py to train SentencePiece on your multilingual corpus, then save it to a directory.


# cd /mnt/db_1/moon/Glot500/tokenization
# cat train.sh        # see what placeholders it has
# # fill in input corpus + output dir, then run it
# bash train.sh
# 2. Build the training corpus (preprocessing/merge_files.sh)
# This concatenates per-language raw text files into the single train.txt that --train_file reads.


# cd /mnt/db_1/moon/Glot500/preprocessing
# cat merge_files.sh
# # fill in input language files + output path
# bash merge_files.sh
# Both of these need the raw Glot500-C language data, which isn't in the repo — you'd download it from wherever the Glot500 authors host it (the README's data section points to the data download instructions).

# One-shot dry-run check (no GPU work)
# Before launching a real run, you can sanity-check the args parse correctly by adding --max_train_samples 100 and running on 1 GPU for a few steps:


# CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python run.py \
#   --model_name_or_path xlm-roberta-base \
#   --train_file       /mnt/db_1/moon/Glot500/data/train.txt \
#   --tokenizer_name   /mnt/db_1/moon/Glot500/tokenization/output \
#   --output_dir       /tmp/glot500-smoke \
#   --cache_dir        /mnt/db_1/moon/Glot500/cache \
#   --per_device_train_batch_size 4 \
#   --fp16 True \
#   --do_train \
#   --max_train_samples 100 \
#   --num_train_epochs 1 \
#   --save_steps 999999
# If that runs end-to-end without crashing, the full distributed launch should also work.

# Optional tuning notes (A6000 has lots of headroom)
# --per_device_train_batch_size 12 is conservative for 48 GB. You can likely go to 24 or 32 with fp16. Adjust --gradient_accumulation_steps proportionally to keep the effective batch where you want it. Watch nvidia-smi during the first few hundred steps to confirm you're not OOMing.
# Consider --bf16 True instead of --fp16 True. A6000 supports bfloat16 natively, and bf16 is more numerically stable for transformer pretraining (no loss-scale issues).
# save_total_limit (e.g. --save_total_limit 5) keeps only the 5 most recent checkpoints — xlm-roberta-base checkpoints are ~1.1 GB each, and at save_steps=10000 × 100 epochs you'll otherwise rack up tens of them.
# Consider --logging_steps 100 so loss curves print every 100 steps (default is every 500).
# TL;DR
# Save the block above as train_base.sh (or edit the existing one in place). It's the upstream recipe adapted to your hardware. The two data/ and tokenization/output paths need real artifacts built first — the rest will be created automatically when you launch.

# Want me to walk through tokenization/train.sh and preprocessing/merge_files.sh next, so you can see what they expect as input?