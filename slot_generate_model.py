from pyexpat import model
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
import os.path

import logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


cache_dir = Path("./cache/slot/")
data_dir = "./data/slot/"
max_len = 128
embedding_dim = 300

with open(cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

embeddings = torch.load(cache_dir / "embeddings.pt")

tag_idx_path = cache_dir / "tag2idx.json"
tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

# Contribution of train data
def main():
    
    tokens = {}
    labels = {}
    sequences = {}
    test_id = []
    test_shape = []

    for which_data in ['train','eval','test']:
        with open(data_dir + f"{which_data}.json") as data_f:
            data = json.load(data_f)
        tokens[which_data] = [info['tokens'] for info in data]
        sequences[which_data] = [vocab.encode(_tokens) for _tokens in tokens[which_data]]
        if which_data != 'test':
            tags_list = [info['tags'] for info in data]
            labels[which_data] = [[tag2idx[tag] for tag in tags] for tags in tags_list]
            labels[which_data] = pad_sequences(labels[which_data], padding='post',
                                            maxlen =max_len,
                                            value =1, dtype ='int32')
            labels[which_data] = to_categorical(labels[which_data].reshape(-1,))
        else:
            test_shape = [len(info['tokens']) for info in data]
            test_id = [info['id'] for info in data]
        sequences[which_data] = pad_sequences(sequences[which_data], padding='post', maxlen =max_len,
                                    value =0, dtype ='int32')
        sequences[which_data] = sequences[which_data].reshape(-1,)
#  Data Type
# tokens : Dict['train','eval'] =  line by line of each sentences, shape = (927242,) 
# sequences : Dict['train', 'eval'] = line by line of each idx-sentences after padding
#         shape = (927232,), class = ndarray
# labels : label, i.e.,  tag -> index of tag(label) -> one hot of index, 
#         e.q. O  -> 1  -> (0,1,0,0,0,0,0,0,0)
#         shape = (927232,9), class = ndarray

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

# Resault : accuracy: 0.9942 - val_loss: 0.0146 - val_accuracy: 0.9941
# Predict test slot
    predictions = bid_model.predict(sequences['test'])
    predictions = np.argmax(predictions, axis=-1)
    predictions = predictions.reshape(len(test_id),-1)

    slot_predictions = {}
    for n in range(len(predictions)):
        slot_predictions[test_id[n]] = predictions[n][:test_shape[n]]     

    idx2slot = {idx:intent for intent,idx in tag2idx.items()}

    for n in range(len(slot_predictions)):
        slot_predictions[test_id[n]] = [idx2slot[idx] for idx in slot_predictions[test_id[n]]]


# Output prediction

    with open('slot_predictions.csv', 'w', encoding='UTF8') as f:
        f.write('id,tags\n')
        for key in slot_predictions.keys():
            f.write("%s"%key)
            c = 1
            for tag in slot_predictions[key]:
                if c != len(slot_predictions[key]):
                    f.write("%s "%(tag))
                else:
                    f.write("%s\n"%(tag))
                c += 1
# Save model
    if os.path.isfile('./slot_Bi_LSTM_model.h5') is False:
        bid_model.save('./slot_Bi_LSTM_model.h5')


if __name__ == "__main__":
    main()