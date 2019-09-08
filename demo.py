from mlhub.pkg import azkey, azrequest, mlask, mlcat

mlcat("Text Classification of MultiNLI Sentences Using BERT", """\
To run through the demo quickly the QUICK_RUN flag
is set to True and so uses a small subset of the data and
a smaller number of epochs.

The table below provides some reference running times on two 
machine configurations for the full dataset.

|QUICK_RUN |Machine Configurations                  |Running time|
|----------|----------------------------------------|------------|
|False     |4 CPUs, 14GB memory                     | ~19.5 hours|
|False     |1 NVIDIA Tesla K80 GPUs, 12GB GPU memory| ~ 1.5 hours|

To avoid CUDA out-of-memory errors the BATCH_SIZE and MAX_LEN are reduced,
resulting in a compromised model performance but one that can be computed
on a typcial user's laptop. For best model performance this same script can
be run on cloud compute (Azure) with the parameters set to their usual values.
""")

QUICK_RUN = True

import sys
sys.path.append("../../")
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

from utils_nlp.dataset.multinli import load_pandas_df
from utils_nlp.eval.classification import eval_classification
from utils_nlp.models.bert.sequence_classification import BERTSequenceClassifier
from utils_nlp.models.bert.common import Language, Tokenizer
from utils_nlp.common.timer import Timer

mlask(end="\n")

mlcat("Introduction", """\
First we will fine-tune and evaluate a pretrained BERT model on a
subset of the MultiNLI dataset.

A sequence classifier is used to wrap Hugging Face's PyTorch
implementation of Google's BERT.
""")

mlask(end="\n")

TRAIN_DATA_USED_FRACTION = 1
TEST_DATA_USED_FRACTION = 1
NUM_EPOCHS = 1

if QUICK_RUN:
    TRAIN_DATA_USED_FRACTION = 0.01
    TEST_DATA_USED_FRACTION = 0.01
    NUM_EPOCHS = 1

if torch.cuda.is_available():
    BATCH_SIZE = 32
else:
    BATCH_SIZE = 8

DATA_FOLDER = "./temp"
BERT_CACHE_DIR = "./temp"
LANGUAGE = Language.ENGLISH
TO_LOWER = True
MAX_LEN = 50 # 150
BATCH_SIZE_PRED = 256 # 512
TRAIN_SIZE = 0.6
LABEL_COL = "genre"
TEXT_COL = "sentence1"

mlcat("Read the Dataset", """\
We start by loading a subset of the data. The following function also
downloads and extracts the files, if they don't exist in the data
folder.

The MultiNLI dataset is mainly used for natural language inference (NLI)
tasks, where the inputs are sentence pairs and the labels are entailment
indicators. The sentence pairs are also classified into _genres_ that
allow for more coverage and better evaluation of NLI models.

For our classification task, we use the first sentence only as the text
input, and the corresponding genre as the label. We select the examples
corresponding to one of the entailment labels (_neutral_ in this case)
to avoid duplicate rows, as the sentences are not unique, whereas the
sentence pairs are.

Below we show the first few sentences from the dataset.
""")

df = load_pandas_df(DATA_FOLDER, "train")
df = df[df["gold_label"]=="neutral"]  # get unique sentences

print(df[[LABEL_COL, TEXT_COL]].head())

mlask(begin="\n", end="\n")

mlcat("Genres", """\
The examples in the dataset are grouped into 5 genres:
""")
print(df[LABEL_COL].value_counts())

mlask(begin="\n", end="\n")

mlcat("Train and Test Datasets", """
We split the data for training and testing, and encode the class labels:
""")

# split

df_train, df_test = train_test_split(df, train_size = TRAIN_SIZE, random_state=0)

df_train = df_train.sample(frac=TRAIN_DATA_USED_FRACTION).reset_index(drop=True)
df_test = df_test.sample(frac=TEST_DATA_USED_FRACTION).reset_index(drop=True)

# encode labels

label_encoder = LabelEncoder()
labels_train = label_encoder.fit_transform(df_train[LABEL_COL])
labels_test = label_encoder.transform(df_test[LABEL_COL])

num_labels = len(np.unique(labels_train))

print("Number of unique labels: {}".format(num_labels))
print("Number of training examples: {}".format(df_train.shape[0]))
print("Number of testing examples: {}".format(df_test.shape[0]))

mlask(begin="\n", end="\n")

mlcat("Tokenize and Preprocess", """\
Before training, we tokenize the text documents and convert them to
lists of tokens. We instantiate a BERT tokenizer given the language, 
and tokenize the text of the training and testing sets.
""")

tokenizer = Tokenizer(LANGUAGE, to_lower=TO_LOWER, cache_dir=BERT_CACHE_DIR)

tokens_train = tokenizer.tokenize(list(df_train[TEXT_COL]))
tokens_test = tokenizer.tokenize(list(df_test[TEXT_COL]))

mlask(begin="\n", end="\n")

mlcat("PreProcessing Steps", """\
We perform the following preprocessing steps:

-   Convert the tokens into token indices corresponding to the BERT
    tokenizer's vocabulary

-   Add the special tokens [CLS] and [SEP] to mark the beginning and end
    of a sentence

-   Pad or truncate the token lists to the specified max length

-   Return mask lists that indicate paddings' positions

-   Return token type id lists that indicate which sentence the tokens
    belong to (not needed for one-sequence classification)


See the original implementation for more information on BERT's input
format.
""")

mlask(end="\n")

tokens_train, mask_train, _ = tokenizer.preprocess_classification_tokens(
    tokens_train, MAX_LEN
)

tokens_test, mask_test, _ = tokenizer.preprocess_classification_tokens(
    tokens_test, MAX_LEN
)

mlcat("Create Model", """\
Next, we create a sequence classifier that loads a pre-trained BERT
model, given the language and number of labels.
""")

classifier = BERTSequenceClassifier(
    language=LANGUAGE, num_labels=num_labels, cache_dir=BERT_CACHE_DIR
)

mlask(end="\n")

mlcat("Train", """\
We train the classifier using the training examples. This involves
fine-tuning the BERT Transformer and learning a linear classification
layer on top of that.
""")

with Timer() as t:
    classifier.fit(
        token_ids=tokens_train,
        input_mask=mask_train,
        labels=labels_train,    
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,    
        verbose=True,
    )    

print("[Training time: {:.3f} hrs]".format(t.interval / 3600))

mlask(begin="\n", end="\n")

mlcat("Score", """\
We now score the test set using the trained classifier.
""")

preds = classifier.predict(token_ids=tokens_test, 
                           input_mask=mask_test, 
                           batch_size=BATCH_SIZE_PRED)

mlask(begin="\n", end="\n")

mlcat("Evaluate Results", """\
Finally, we compute the accuracy, precision, recall, and F1 metrics of
the evaluation on the test set.
""")

report = classification_report(labels_test, preds, target_names=label_encoder.classes_, output_dict=True) 
accuracy = accuracy_score(labels_test, preds )
print("accuracy: {:.2}".format(accuracy))
# TODO Iterate through the json and print it properly!!!
print(json.dumps(report, indent=4, sort_keys=True))

mlask(begin="\n")

