from msilib import sequence
import pandas as pd
import csv
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import torch
from tqdm import trange
from utils import Vocab
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Bidirectional, Embedding, LSTM, Dense, Dropout,GlobalMaxPool1D
from tensorflow.keras.models import Model

# Check GPU support 
# tf.test.is_gpu_available(
#     cuda_only=False, min_cuda_compute_capability=None
# )
# output = True

cache_dir = Path("./cache/slot/")
data_dir = "./data/slot/"

max_len = 128
embedding_dim = 300

# Contribution of train data
with open(cache_dir / "vocab.pkl", "rb") as f:
    vocab: Vocab = pickle.load(f)

embeddings = torch.load(cache_dir / "embeddings.pt")

tag_idx_path = cache_dir / "tag2idx.json"
tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

tokens = {}
labels = {}
sequences = {}

# train_tokens =[]
# train_tags =[]
# train_sequences =[]
# val_tokens =[]
# val_tags =[]
# val_sequences =[]

for which_data in ['train','eval']:
    with open(data_dir + f"{which_data}.json") as data_f:
        data = json.load(data_f)

    tokens[which_data] = [info['tokens'] for info in data]
    sequences[which_data] = [vocab.encode(_tokens) for _tokens in tokens[which_data]]
    tags_list = [info['tags'] for info in data]
    labels[which_data] = [[tag2idx[tag] for tag in tags] for tags in tags_list]

    sequences[which_data] = pad_sequences(sequences[which_data], padding='post', maxlen =max_len,
                                value =0, dtype ='int32')
    labels[which_data] = pad_sequences(labels[which_data], padding='post', maxlen =max_len,
                                value =1, dtype ='int32')

    sequences[which_data] = sequences[which_data].reshape(-1,)
    labels[which_data] = to_categorical(labels[which_data].reshape(-1,))


''' Data Type
tokens : Dict['train','eval'] =  line by line of each sentences 
sequences : Dict['train', 'eval'] = line by line of each idx-sentences after padding
        shape = (927232,), class = ndarray
labels : label, tag -> index of tag(label) -> one hot of index, 
        shape = (927232,9), class = ndarray
'''


# Create the model
sequence_input = Input(shape=(1,), dtype='int32')
embedding_layer = Embedding(embeddings.shape[0],
                            embedding_dim,
                           weights = [embeddings],
                           input_length = 1,
                           trainable=False,
                           name = 'embeddings')

embedded_sequences = embedding_layer(sequence_input)

x = Bidirectional(LSTM(90, 
        return_sequences=True,
        name='lstm_layer',
        go_backwards=True))(embedded_sequences)

x = Dropout(0.1)(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
preds = Dense(9, activation="sigmoid")(x)


bid_model = Model(sequence_input, preds)
bid_model.compile(loss = 'categorical_crossentropy',
             optimizer='adam',
             metrics = ['accuracy'])
bid_model.summary()


# Model training
history = bid_model.fit(sequences['train'], labels['train'], 
                    epochs =15, batch_size =64, 
                    validation_data=(sequences['train'], labels['train']))
# Resault :loss: 0.0690 - accuracy: 0.9790 - val_loss: 0.4592 - val_accuracy: 
# 0.9057
prediction = bid_model.predict(train_sequences)
# Presict test data
test_text =[]
test_id =[]
test_sequences =[]
with open(data_dir + "test.json") as data_f:
    data = json.load(data_f)

for info in data:
    test_text.append(info['text'])
    test_id.append(info['id'])
    vector = vocab.encode(info['text'].split(' '))
    test_sequences.append(vector)
test_sequences = pad_sequences(test_sequences, padding='post', maxlen = max_len)

predictions = bid_model.predict(test_sequences)
idx_predictions = np.argmax(predictions, axis=-1)
intent_predictions = {}
idx2intent = {idx:intent for intent,idx in intent2idx.items()}
for n in range(len(idx_predictions)):
    intent_predictions[test_id[n]] = idx2intent[idx_predictions[n]]

import csv
with open('predictions.csv', 'w', encoding='UTF8') as f:
    f.write('id,intent\n')
    for key in intent_predictions.keys():
        f.write("%s,%s\n"%(key,intent_predictions[key]))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )
    # data
    parser.add_argument("--max_len", type=int, default=128)
    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    # data loader
    parser.add_argument("--batch_size", type=int, default=128)
    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
