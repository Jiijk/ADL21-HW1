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

cache_dir = Path("./cache/intent/")
data_dir = "./data/intent/"

max_len = 128
embedding_dim = 300
TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

# Contribution of train data

with open(cache_dir / "vocab.pkl", "rb") as f:
    vocab: Vocab = pickle.load(f)
embeddings = torch.load(cache_dir / "embeddings.pt")

train_text =[]
train_labels =[]
train_sequences =[]
with open(data_dir + "train.json") as data_f:
    data = json.load(data_f)

for info in data:
    train_text.append(info['text'])
    vector = vocab.encode(info['text'].split(' '))
    train_sequences.append(vector)
    train_labels.append(info['intent'])
train_sequences = pad_sequences(train_sequences, padding='post', maxlen = max_len)

intent_idx_path = cache_dir / "intent2idx.json"
intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())
for n in range(len(train_labels)):
    train_labels[n] = intent2idx[train_labels[n]]
train_labels = to_categorical(np.asarray(train_labels))

# Contribution of validation data
val_text =[]
val_labels =[]
val_sequences =[]
with open(data_dir + "eval.json") as data_f:
    data = json.load(data_f)

for info in data:
    val_text.append(info['text'])
    vector = vocab.encode(info['text'].split(' '))
    val_sequences.append(vector)
    val_labels.append(info['intent'])
val_sequences = pad_sequences(val_sequences, padding='post', maxlen = max_len)

for n in range(len(val_labels)):
    val_labels[n] = intent2idx[val_labels[n]]
val_labels = to_categorical(np.asarray(val_labels))


''' Data Type
train_text : list , shape= (15000,)
train_seqences : np.array, shape = (15000, 128) ,將text出現的word按照出現頻率來編號，
                編號方式同vocab
train_labels : list, shape = (15000,)
'''


# Create the model
sequence_input = Input(shape=(max_len,), dtype='int32')
embedding_layer = Embedding(embeddings.shape[0],
                            embedding_dim,
                           weights = [embeddings],
                           input_length = max_len,
                           trainable=False,
                           name = 'embeddings')

embedded_sequences = embedding_layer(sequence_input)

'''
LSTM without dropout
'''
x = LSTM(60, 
        return_sequences=True,
        name='lstm_layer',
        go_backwards=True)(embedded_sequences)

x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
preds = Dense(150, activation="sigmoid")(x)

model = Model(sequence_input, preds)
model.compile(loss = 'categorical_crossentropy',
             optimizer='adam',
             metrics = ['accuracy'])
model.summary()
history = model.fit(train_sequences, train_labels, 
                    epochs = 15, batch_size=32, 
                    validation_data=(val_sequences, val_labels))
# Resaul : accuracy: 0.9918 - val_loss: 0.6207 - val_accuracy: 
# 0.8657

'''
LSTM
'''
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
preds = Dense(150, activation="sigmoid")(x)

model = Model(sequence_input, preds)
model.compile(loss = 'categorical_crossentropy',
             optimizer='adam',
             metrics = ['accuracy'])
model.summary()


print('Training progress:')
history = model.fit(train_sequences, train_labels, 
                    epochs = 15, batch_size=32, 
                    validation_data=(val_sequences, val_labels))
# Resault : loss: 0.1407 - accuracy: 0.9581 - val_loss: 0.5256 - val_accuracy: 
# 0.8873

'''
Bidirection with LSTM
'''
x = Bidirectional(LSTM(60, 
        return_sequences=True,
        name='lstm_layer',
        go_backwards=True))(embedded_sequences)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
preds = Dense(150, activation="sigmoid")(x)

bid_model = Model(sequence_input, preds)
bid_model.compile(loss = 'categorical_crossentropy',
             optimizer='adam',
             metrics = ['accuracy'])
bid_model.summary()

# Model training
print('Training progress:')
history = bid_model.fit(train_sequences, train_labels, 
                    epochs = 15, batch_size=32, 
                    validation_data=(val_sequences, val_labels))
# Resault :loss: 0.0690 - accuracy: 0.9790 - val_loss: 0.4592 - val_accuracy: 
# 0.9057

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
