import argparse, sys, logging, copy, os, json, codecs
from datetime import datetime
import sentencepiece as spm

import sentencepiece_model_pb2 as sp_model

from os import listdir

from transformers import XLMRobertaTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fname', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--save_directory', type=str)
    parser.add_argument('--vocab_size', type=int)

    args = parser.parse_args()
    
    #Get new tokens
    input_fname = args.input_fname
    output_fname = args.save_directory + 'Glot500'
    cmd = ('--input={} --model_prefix={} --vocab_size={} '
           '--model_type=unigram --character_coverage=1.0 '
           '--train_extremely_large_corpus=true '
           '--input_sentence_size=20000000 --shuffle_input_sentence=true '
           '--num_threads=32').format(input_fname, output_fname, args.vocab_size)
    
    #MJ: 1)The default behavior (no flag) is "ingest every line of --input". For Glot500.txt that meant ~698 M sentences materialized in RAM during the Normalizing sentences... stage, plus an even larger candidate-substring index built on top during the unigram seed step. That's what blew past 300 GiB and aborted.
    # With this flag set to N, the trainer reads the input file and stops loading after it has retained N sentences in memory. Everything downstream — normalization, suffix-array construction, EM iterations, pruning — operates only on those N sentences. The peak working set scales roughly linearly with N, so going from 698 M → 20 M cuts memory ~35×.
    # Why 20 million is enough. Unigram language model training learns a probability distribution over candidate subword pieces by counting their occurrences in the sample. The variance of those counts shrinks as √N — so beyond a few million sentences, additional data buys almost no improvement in piece quality. The Glot500 authors, and most multilingual SPM recipes, sample on the order of 10-50 M sentences from corpora of this scale. With 250 K target pieces and ~500 languages, 20 M gives ~40 000 sentences per language on average — comfortably more than enough.
    # 2) With --shuffle_input_sentence=true, SP uses proper reservoir sampling: as it streams through the whole file, each new sentence has equal probability of replacing a random slot in the in-memory buffer. The end result is a uniform random sample of N sentences drawn from the entire file, regardless of file order or length.
    # The trade-off is that SP still has to scan the entire file to do the random sampling correctly (the first-N strategy could stop reading early)
    # => Together, --input_sentence_size=20000000 --shuffle_input_sentence=true mean: "Scan the full corpus, draw 20 M sentences uniformly at random, throw away the rest, and train on the sample."



    spm.SentencePieceTrainer.train(cmd)

    #Load pretrained XLM-R SPM:
    #MJ: Why use sentencepiece_model_pb2 and not the spm.SentencePieceProcessor API?
    # The high-level SentencePieceProcessor is read-only — you can encode/decode but not mutate the vocabulary. 
    # To add pieces to an existing model you have to edit the underlying protobuf, which is exactly what sentencepiece_model_pb2.ModelProto exposes. That's the whole reason this file is vendored into the repo

    original_m = sp_model.ModelProto()
    original_m.ParseFromString(open(args.save_directory + 'sentencepiece.bpe.model', 'rb').read())
    new_m = sp_model.ModelProto()
    new_m.ParseFromString(open(output_fname + '.model', 'rb').read())

    add_cnt = 0 
    piece_d = {piece.piece: 0 for piece in original_m.pieces}
    for new_piece in new_m.pieces:
        if new_piece.piece not in piece_d:
            piece_to_add = sp_model.ModelProto().SentencePiece()
            # Add token
            piece_to_add.piece = new_piece.piece
            # Add token log-prob
            piece_to_add.score = new_piece.score
            original_m.pieces.append(piece_to_add)
            add_cnt += 1

    print('Add {} tokens'.format(add_cnt)) #MJ: 165727
    logging.info('Add {} tokens'.format(add_cnt))
    
    new_spm_save_dir = args.save_directory + 'Glot500_extended_spm.model'
    with open(new_spm_save_dir, 'wb') as f:
        f.write(original_m.SerializeToString())
    
    tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name)
    tokenizer.vocab_file = new_spm_save_dir
    tokenizer.sp_model.load(tokenizer.vocab_file)
    tokenizer.save_pretrained(args.save_directory + 'Glot500_extended_spm')

    print('Tokenizer training is done. Wrote {}'.format(
        args.save_directory + 'Glot500_extended_spm'))

