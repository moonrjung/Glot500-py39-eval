#MJ: axi1500 is a zero-shot classification task: train on English, evaluate on other languages.
#  The English half is freely shippable. But the other-language test sets are derived 
# from the Parallel Bible Corpus (PBC) — a collection of Bible translations across 1500+ languages, 
# maintained by Michael Cysouw at Philipps University of Marburg. Most of those bibles are under copyright, 
# so the Taxi1500 authors can't redistribute the processed data; 
# instead they release just the raw "corpus" indices + data_preprocess.py, and 
# tell you to email Cysouw to get the underlying bibles (academic use only).

# What you can do right now:

# ✅ Train and evaluate on English-only with the three TSVs we're about to download 
# — gives you a working pipeline and a sanity-check number.

# Real zero-shot evaluation across 354 language-scripts needs PBC. To unlock that:
# Email Michael Cysouw at Marburg (the Taxi1500 README has the contact line) requesting PBC access 
# for academic use.
# Once you have the corpus, clone cisnlp/Taxi1500, drop the PBC into the expected layout, 
# and run data_preprocess.py 
# — it produces per-language test TSVs.
# Adapt zero_shot_train.py to loop over those per-language test files (it currently references 
#     an undefined test_file at zero_shot_train.py:91).

#MJ: So the script is doing fine-tuning: a pretrained MLM backbone + a new linear classifier, 
# trained end-to-end on Taxi1500 English data. No local pretraining anywhere.

import torch
import pandas as pd
import transformers
from transformers import AdamW
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
import numpy as np
import random
from sklearn.metrics import f1_score
from collections import Counter
import time
import datetime
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels

from packaging.version import parse as _v

GLOT500_PATH = "/home/sogang/mnt/db_1/models/glot500-base"


seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

epochs = 30
batch_size = 8

accum_iter = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

best_model='best_model.pt'

#MJ: The cisnlp/Taxi1500 GitHub repo has a folder named eng_data/ containing three pre-built files:

# eng_train.tsv — English training set (verses + class labels)
# eng_dev.tsv — English dev/validation set
# eng_test.tsv — English test set

train_file = "/home/sogang/mnt/db_1/moon/Glot500/evaluation/download_data/download/taxi1500/eng_train.tsv"
dev_file   = "/home/sogang/mnt/db_1/moon/Glot500/evaluation/download_data/download/taxi1500/eng_dev.tsv"
test_file  = "/home/sogang/mnt/db_1/moon/Glot500/evaluation/download_data/download/taxi1500/eng_test.tsv"

# Change line 64 to point at a non-English test file of the same six-class schema, e.g.


# test_file = "deu_test.tsv"   # or whatever per-language file you have
# You'd want either:

# a loop over multiple language files (the paper-style evaluation across all 354 language-scripts), or
# a CLI flag so you can sweep externally.
# Both require the per-language test TSVs, which require PBC access + running data_preprocess.py from cisnlp/Taxi1500.





def read_data(file):
    print('Preparing dataset...')
#     #MJ: (requires pandas ≥1.3)
#     # df = pd.read_csv(file, delimiter='\t', header=None,on_bad_lines='skip', engine='python')

#     #MJ: on pandas 1.1.5)
#     df = pd.read_csv(file, delimiter='\t', header=None,
#                  error_bad_lines=False, warn_bad_lines=False, engine='python')
#     df.columns = ['id', 'label', 'verse']
#     print('Number of sentences: {:,}\n'.format(df.shape[0]))
#     #print(Counter(list(df['label'])))
#    # print(len(Counter(list(df['label']))))
#     df['label'].replace({'Recommendation': int(0), 'Faith': int(1), 'Violence': int(2), 'Grace': int(3), 'Sin': int(4), 'Description': int(5)},inplace=True)


   
    _kw = {'on_bad_lines': 'skip'} if _v(pd.__version__) >= _v('1.3') \
              else {'error_bad_lines': False, 'warn_bad_lines': False}

    df = pd.read_csv(file, delimiter='\t', header=None, engine='python', **_kw)

    df.columns = ['id', 'label', 'verse']
    print('Number of sentences: {:,}\n'.format(df.shape[0]))
    print(Counter(list(df['label'])))
    print(len(Counter(list(df['label']))))

    df['label'].replace({'Recommendation': int(0), 'Faith': int(1), 'Violence': int(2), 'Grace': int(3), 'Sin': int(4), 'Description': int(5)},inplace=True)


    sentences = df.verse.values
    labels = df.label.values
    #print(list(df['label']))
    #print(len(sentences), len(labels))
    return sentences, labels

def encode(sentences, labels):
    print('Loading model tokenizer...')
    input_ids = []
    attention_masks = []

    #MJ: from_pretrained API, which fetches the published checkpoint from the Hub the first time it's called.

    tokenizer=AutoTokenizer.from_pretrained(GLOT500_PATH)
    for sent in sentences:
        sent = str(sent)
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            truncation=True,
            max_length=100,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    labels = torch.tensor(labels)

        #print('Original: ', sentences[0])
        #print('Token IDs:', input_ids[0])

    return input_ids, attention_masks, labels


#MJ: 6 Labels:
# Recommendation is prescriptive — commands, exhortations, "thou shalt", moral advice. ("Honor thy father and thy mother.")
# Faith is devotional — belief, trust, prayer, religious practice. ("Blessed are those who have not seen and yet have believed.")
# Violence is destructive action — battles, killings, divine judgment in physical form. ("They smote them with the edge of the sword.")
# Grace is divine giving — God's unearned favor and blessing. ("For by grace are ye saved through faith.")
# Sin is transgression — wrongdoing, disobedience, evil acts. ("The wages of sin is death.")
# Description is the neutral catch-all — narrative, genealogies, geography, who-said-what, with no overt moral or theological content. ("And Abraham journeyed from thence toward the south country.")


#import pdb; pdb.set_trace()

train_verse, train_label = read_data(train_file)
train_input_ids, train_attention_masks, train_labels = encode(train_verse, train_label)
#MJ: Encode: trings → integer IDs (the core job).
# 1. Tokenization: "For God so loved the world" becomes something like [0, 717, 2362, 2071, 21603, 70, 8999, 2, 0, 0, ...]. The model has no notion of characters or words — only token IDs that index into its embedding table. Without this step you literally cannot feed text to the transformer.

# 2. Variable length → fixed length (100).
# Verses are different lengths, but a batch tensor must be rectangular. So shorter verses are padded with a special <pad> ID up to 100; longer ones are truncated. This is what makes batching possible.

# 3. The attention mask falls out of step 2.
train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)

dev_verse, dev_label = read_data(dev_file)
dev_input_ids, dev_attention_masks, dev_labels = encode(dev_verse, dev_label)
val_dataset = TensorDataset(dev_input_ids, dev_attention_masks, dev_labels)


test_verse, test_label = read_data(test_file)
test_input_ids, test_attention_masks, test_labels = encode(test_verse, test_label)
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)

train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size,# Trains with this batch size.
       )

validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

test_dataloader=DataLoader(
    test_dataset,
    sampler = SequentialSampler(test_dataset),
    batch_size = batch_size,
)

# MJ: (1) What gets loaded vs. what gets newly initialized
# cis-lmu/glot500-base is an XLM-RoBERTa-style MLM checkpoint. 
# Calling AutoModelForSequenceClassification.from_pretrained(...) with num_labels=6 does this:

# Component	Source
# Token embeddings + positional embeddings	✅ Loaded from the checkpoint
# 12 Transformer encoder layers (attention + FFN)	✅ Loaded from the checkpoint
# MLM head (lm_head — projects back to vocab for masked-token prediction)	❌ Discarded (not needed for classification)
# Classification head (the new top)	🆕 Newly initialized with random weights
# The classification head, for XLM-RoBERTa, is a small 2-layer module:

# class XLMRobertaClassificationHead(nn.Module):
#     self.dense    = nn.Linear(hidden_size, hidden_size)   # 768 → 768
#     self.dropout  = nn.Dropout(...)
#     self.out_proj = nn.Linear(hidden_size, num_labels)    # 768 → 6

#MJ: (2) When train(...) runs:

# All parameters get gradients by default — backbone and the new head.
# The new classifier.* weights start from random and move a lot (large gradients).
# The backbone weights start from pretrained values and move a little (small lr=2e-5,
#          mostly refining cross-lingual features the MLM pretraining already learned).
# This "frozen-init head on top of a pretrained encoder, then update everything together" is 
# the standard BERT-family fine-tuning recipe. If you wanted a "linear probe" 
# instead (freeze backbone, only train head), 
# you'd have to explicitly do for p in model.roberta.parameters(): p.requires_grad = False —'
# ' but the Glot500 paper's setup, and this script, both fine-tune end-to-end.


model = AutoModelForSequenceClassification.from_pretrained(
    GLOT500_PATH, # local checkpoint dir; was 'cis-lmu/glot500-base' before transformers 4.18 + HF API mismatch
    num_labels = 6, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)
model.cuda()

optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

total_steps = len(train_dataloader) * epochs

# With num_warmup_steps=0, the LR starts at 2e-5 defined in Adam on step 0 and linearly decays to 0 by the last step. 
# So "small lr" is actually "small peak lr that further decays toward zero" —
# the backbone moves even less in later epochs.

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def F1_score(preds, labels):
    pre = np.argmax(preds, axis=1)
    f1 = f1_score(labels, pre, average='macro')
    return f1

def train(train_dataloader, validation_dataloader):
    training_stats = []
    best_loss = float('inf')
    best_acc = 0
    nb_eval_steps = 0

    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        total_train_loss = 0
        total_train_f1 = 0

        model.cuda()
        model.train()
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()

            with torch.set_grad_enabled(True):

                (loss, logits) = model(b_input_ids,
                                    attention_mask=b_input_mask,
                                    labels=b_labels,
                                    return_dict=False)

                logits = logits.detach().cpu().numpy()
                b_label_ids = b_labels.to('cpu').numpy()

                total_train_loss += loss.item()
                total_train_f1 += F1_score(logits, b_label_ids)

                prediction = np.argmax(logits, axis=1)
                #print('train predict:', prediction)
                #print('true_label', b_label_ids)
                loss = loss / accum_iter
                loss.backward() #MJ: Accumulate the gradients every accum_iter steps
                if ((step + 1) % accum_iter == 0) or (step + 1 == len(train_dataloader)):
                    #MJ: Update the parameter theta of the network every accum_iter:

                    optimizer.step()
                    scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_f1 = total_train_f1 / len(train_dataloader)
        print('train_f1: ', avg_train_f1)

        model.eval()

        total_eval_accuracy = 0
        total_eval_f1_score = 0
        total_eval_loss = 0
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            with torch.no_grad():
                (loss, logits) = model(b_input_ids,
                                       attention_mask=b_input_mask,
                                       labels=b_labels,
                                       return_dict=False)
            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_f1_score += F1_score(logits, label_ids)
            total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        avg_val_f1 = total_eval_f1_score / len(validation_dataloader)
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        print('avg_val_accuracy:', avg_val_accuracy, 'avg_val_f1: ', avg_val_f1, 'avg_val_loss', avg_val_loss)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model, best_model)
            print('loss descreased:', epoch_i)
        else:
            pass
        print('val_f1: ', avg_val_f1)

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Train Loss': avg_train_loss,
                'Val Loss': avg_val_loss,
                'Val Accu.': avg_val_accuracy,
                'val F1 ': avg_val_f1,
            }
        )

        print("")
        print("Training complete!")
    pd.set_option('display.precision', 2)
    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')
    print(df_stats)


def test(test_dataloader):
    print('Start Test...')
    model = torch.load(best_model)
    model.cuda()
    model.eval()
    predictions, true_labels = [], []
    #prediction_list, actual_label=np.array(), np.array()

    total_test_accuracy = 0
    total_test_f1_score = 0
    #total_test_loss = 0
    nb_eval_steps = 0
    # Predict
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids,
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        #MJ: F1_c = 2 · P_c · R_c / (P_c + R_c) ← per-class F1 
        #F1 needs TP/FP/FN, which require a binary lens. Multi-class classification supplies that lens automatically
        #  by treating each class as its own yes/no problem. The model still outputs one of six labels — but the metric views that single 6-way prediction
        #  as six simultaneous yes/no calls under the hood. That's how a metric born in retrieval ports cleanly to multi-class classification.

        total_test_f1_score += F1_score(logits, label_ids)
        total_test_accuracy += flat_accuracy(logits, label_ids)
        # Store predictions and true labels

        predictions.append(np.argmax(logits, axis=1).flatten())

        true_labels.append(label_ids)

    avg_test_accuracy = total_test_accuracy / len(true_labels)
    #MJ: Macro-F1 = (F1_0 + F1_1 + F1_2 + F1_3 + F1_4 + F1_5) / 6     ← the average
    avg_test_f1 = total_test_f1_score / len(true_labels)

    print("  Accuracy: {0:.2f}".format(avg_test_accuracy))
    print("  F1: {0:.2f}".format(avg_test_f1))
    print('    DONE.')

if __name__ == "__main__":

    train(train_dataloader, validation_dataloader)
    test(test_dataloader)







