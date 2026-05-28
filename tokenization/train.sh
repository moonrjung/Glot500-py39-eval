#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python run.py --input_fname    /home/sogang/mnt/db_1/moon/Glot500/data/Glot500.txt \
  --model_name     xlm-roberta-base \
  --save_directory /home/sogang/mnt/db_1/moon/Glot500/tokenization/output/ \
  --vocab_size     250000

##python -m debugpy --listen 5678 --wait-for-client run.py \
# Prerequisites:  Create the output directory and drop XLM-R's pretrained SPM into it
# The script reads {save_directory}sentencepiece.bpe.model as the base XLM-R tokenizer to extend. 
# 1) You have to put it there yourself — easiest way is via the HuggingFace cache:


# mkdir -p /home/sogang/mnt/db_1/moon/Glot500/tokenization/output

# 2)  Pull xlm-roberta-base's tokenizer files (downloads to HF cache) and copy the SPM
# python -c "
# from transformers import XLMRobertaTokenizer
# tok = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
# print('SPM file is at:', tok.vocab_file)
# "

# # That will print something like:
#SPM file is at: /home/sogang/.cache/huggingface/hub/models--xlm-roberta-base/snapshots/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089/sentencepiece.bpe.model


# # Copy it into the save_directory (replace <SRC> with the printed path):
# cp <SRC> /home/sogang/mnt/db_1/moon/Glot500/tokenization/output/sentencepiece.bpe.model

# Alternatively, one-liner that bypasses the cache lookup:


# mkdir -p /disk3/moon/Glot500/tokenization/output

# curl -L \
#   https://huggingface.co/xlm-roberta-base/resolve/main/sentencepiece.bpe.model \
#   -o /disk3/moon/Glot500/tokenization/output/sentencepiece.bpe.model
