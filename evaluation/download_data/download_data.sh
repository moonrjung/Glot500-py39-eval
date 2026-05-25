#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

REPO=$PWD #MJ: $PWD=Glot500/evaluation/download_data
DIR=$REPO/download
#MJ: mkdir -p $DIR

# Helper function to download the UD-POS data.
# In order to ensure backwards compatibility with the XTREME evaluation,
# languages in XTREME use the UD version used in the original paper; for the new
# languages in XTREME-R, we use a more recent UD version.

#MJ: called by  download_treebank all $base_dir $out_dir:
# base_dir=$DIR/udpos-tmp
# out_dir=$base_dir/conll/
#MJ:  Universal Dependencies treebanks

#MJ: each UD treebank corresponds to one specific corpus in one language, but a language can have multiple treebanks.

#MJ: UD is a project to provide consistent morphological and syntactic annotations for many languages.
# Each "treebank" is a corpus of sentences with annotations including:

# POS tags (part of speech: NOUN, VERB, ADJ, ...)
# Morphological features (case, number, tense, ...)
# Syntactic dependency trees

#MJ: UD covers ~100+ languages and is the de facto standard for cross-lingual POS tagging and dependency parsing evaluation. 
# Each treebank file is in the CoNLL-U format (Conference on Natural Language Learning, "Universal" extension) — 
# a tab-separated format with one token per line and several annotation columns.


function download_treebank {
    base_dir=$2  #MJ: base_dir=$DIR/udpos-tmp;    
    out_dir=$3    #MJ:  out_dir=$base_dir/conll/

    ud_version="2.11"
    tarball=$base_dir/ud-treebanks-v$ud_version.tgz
    #MJ: lindat migrated to DSpace 7; the old /xmlui/bitstream/handle/... URL now returns
    #MJ: an HTML landing page with HTTP 200, which --fail does not catch. The handle
    #MJ: hdl:11234/1-4923 still resolves; this is its current bitstream content URL.
    #MJ: If lindat re-uploads and this UUID 404s, refresh via:
    #MJ:   curl -sI "https://lindat.mff.cuni.cz/repository/server/api/pid/find?id=hdl%3A11234%2F1-4923"
    #MJ:   -> Location: .../items/<item-uuid>; then GET .../items/<item-uuid>/bundles
    #MJ:   -> ORIGINAL bundle's bitstreams list -> uuid for ud-treebanks-v2.11.tgz
    url=https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams/b3e3a82b-7c85-489d-904a-c93c155b53a6/content
    echo "$url"
    curl -L --fail -o "$tarball" "$url"
    #MJ: -L follows redirects, --fail bails on HTTP errors,
    #MJ: -o sets the output filename (URL ends in /content, not in a real filename).

    #MJ: belt-and-braces: DSpace can serve HTML landing pages with HTTP 200, so verify gzip magic
    if ! file "$tarball" | grep -q 'gzip compressed'; then
        echo "ERROR: $tarball is not gzip; download likely returned an HTML page." >&2
        return 1
    fi

    tar -xzf "$tarball"

    #MJ: The script iterates over every .conllu file in every treebank directory
    # ach treebank has its own directory in the UD distribution, with files like:

        # xx_yy-ud-train.conllu — training set
        # xx_yy-ud-dev.conllu — development/validation set
        # xx_yy-ud-test.conllu — test set


    for x in $base_dir/ud-treebanks-v$ud_version/*/*.conllu; do
        file="$(basename $x)" #MJ: After file="$(basename $x)", $file becomes just en_ewt-ud-train.conll
        IFS='_' read -r -a array <<< "$file". #MJ: — IFS is set ONLY for the read command.

        lang=${array[0]}
        lang_dir=$out_dir/$lang/. #MJ: udpos-tmp/conll/en
        mkdir -p $lang_dir #MJ: Creates a per-language output directory ($out_dir/en/).
        y=$lang_dir/${file/conllu/conll} #MJ: Replacing the substring conllu with conll
        
        if [ ! -f "$y" ]; then
            echo "python $REPO/third_party/ud-conversion-tools/conllu_to_conll.py $x $y --lang $lang --replace_subtokens_with_fused_forms --print_fused_forms"
            python $REPO/third_party/ud-conversion-tools/conllu_to_conll.py $x $y --lang $lang --replace_subtokens_with_fused_forms --print_fused_forms
        else
            echo "${y} exists"
        fi
    done
}

# Download UD-POS tagging dataset.
function download_udpos {
    base_dir=$DIR/udpos-tmp
    out_dir=$base_dir/conll/
    #MJ: Skip the ~5GB fetch+unpack+convert if $DIR/udpos already exists. We guard on
    #MJ: the persistent artifact (not $base_dir) because the block ends with
    #MJ: rm -rf $base_dir, so guarding on $base_dir would re-download every run.
    if [ ! -d $DIR/udpos ]; then
        mkdir -p $out_dir
        cd $base_dir
        #MJ:  Universal Dependencies treebanks: UD v2.11 has 228 treebanks across 130+ languages
        download_treebank all $base_dir $out_dir

        cd $REPO
        #MJ: utils_preprocess.py --task udpos then decides which treebank(s) to use for each
        #MJ: language's train/dev/test splits. Glot500 (following XTREME) uses one canonical
        #MJ: treebank per language for consistency — often the largest (EWT for English, GSD for many others).
        python $REPO/utils_preprocess.py --data_dir $out_dir/ --output_dir $DIR/udpos/ --task udpos
        rm -rf $base_dir #MJ: mirrors download_panx; udpos-tmp is scratch (~5GB) after udpos/ is built
        echo "Successfully downloaded data at $DIR/udpos" >> $DIR/download.log
    fi
    python $REPO/utils_preprocess.py --data_dir $DIR/udpos/ --output_dir $DIR/pos/ --task tagging_iso_convert
    cp pos_labels.txt $DIR/pos/labels.txt
}

function download_panx {
    echo "Download panx NER dataset"
    #MJ: So PAN-X is essentially "WikiAnn as packaged for cross-lingual NER evaluation in XTREME."

    base_dir=$DIR/panx_dataset
    if ! [ -d $base_dir ]; then
        # PAN-X / WikiAnn used to ship as AmazonPhotos.zip via an Amazon Photos
        # share link that is no longer reachable. Pull from HuggingFace instead
        # and emit the legacy per-language CoNLL files the preprocessor wants.
        mkdir -p $base_dir
        python $REPO/download_panx_hf.py \
            --lang_list $REPO/ner_lang_list.txt \
            --output_dir $base_dir
        cd $REPO
        python $REPO/utils_preprocess.py \
            --data_dir $base_dir \
            --output_dir $DIR/panx \
            --task panx
        rm -rf $base_dir
        echo "Successfully downloaded data at $DIR/panx" >> $DIR/download.log
    fi
    python $REPO/utils_preprocess.py --data_dir $DIR/panx/ --output_dir $DIR/ner/ --task tagging_iso_convert
    cp ner_labels.txt $DIR/ner/labels.txt
}

function download_tatoeba {
    base_dir=$DIR/tatoeba/
    if ! [ -d $base_dir ]; then
        wget https://github.com/facebookresearch/LASER/archive/main.zip
        unzip -qq -o main.zip -d $base_dir/
        mv $base_dir/LASER-main/data/tatoeba/v1/* $base_dir/
    fi
    echo "Successfully downloaded data at $DIR/tatoeba" >> $DIR/download.log
    python $REPO/utils_preprocess.py --data_dir $DIR/tatoeba/ --output_dir $DIR/retrieval_tatoeba/ --task retrieval_iso_convert
}

# The shell script download_taxi1500 downloads three small English TSV files 
# from the Taxi1500 GitHub repository:
# What eng_data/ actually contains
# The eng_data/ directory on cisnlp/Taxi1500 has three files:
# eng_data/
# ├── eng_train.tsv    # ~860 English Bible verses + labels
# ├── eng_dev.tsv      # ~106 English Bible verses + labels
# └── eng_test.tsv     # ~111 English Bible verses + labels
# Total: 1,067 English Bible verses, each tagged with one of 6 topic categories (Acts, Faith, Praise, Recommendation, Sin, Thanks).
# Each row in each TSV is:
# verse_id    text    label
# 40005003    Blessed are the poor in spirit: for theirs is the kingdom of heaven.    Praise
# 46013004    Charity suffereth long, and is kind; charity envieth not.    Recommendation
# 
# The complete English Bible (Old + New Testaments combined) contains approximately 31,000 verses. Taxi1500 uses only 1,067 of them. Why so few?
# 

function download_taxi1500 {
    echo "Download Taxi1500 text-classification dataset (English split only)"
    base_dir=$DIR/taxi1500
    mkdir -p $base_dir
    #MJ: cisnlp/Taxi1500 ships ready-made English train/dev/test TSVs under eng_data/.
    #MJ: Per-language test sets are derived from the Parallel Bible Corpus (PBC), which
    #MJ: is not redistributable; request academic access from Michael Cysouw (Marburg),
    #MJ: then run cisnlp/Taxi1500's data_preprocess.py to materialise the other languages.
    base_url=https://raw.githubusercontent.com/cisnlp/Taxi1500/main/eng_data
    for f in eng_train.tsv eng_dev.tsv eng_test.tsv; do
        out=$base_dir/$f
        curl -L --fail -o "$out" "$base_url/$f" || { echo "ERROR: failed to fetch $f" >&2; return 1; }
    done
    echo "Successfully downloaded data at $DIR/taxi1500" >> $DIR/download.log
    echo "NOTE: only English split present; per-language test data requires PBC access (see https://github.com/cisnlp/Taxi1500)" >> $DIR/download.log
}

download_tatoeba #MJ: sentence retreival using Tatoeba dataset
download_panx #MJ: NER using PAN-X dataset derived from WikiAnn dataset
download_udpos #MJ: POS Tagging using UD(Universal Dependencies dataset
download_taxi1500 #MJ: Text classification using Taxi1500 (English split only; per-language needs PBC)

