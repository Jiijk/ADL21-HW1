import json
import pickle
from pathlib import Path
from typing import Dict
import torch
from tqdm import trange
from utils import Vocab
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import sys
import os.path

import logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Global Variable
cache_dir = Path("./cache/slot/")
data_dir = "./data/slot/"
max_len = 128
embedding_dim = 300

with open(cache_dir / "vocab.pkl", "rb") as f:
    vocab: Vocab = pickle.load(f)

embeddings = torch.load(cache_dir / "embeddings.pt")

tag_idx_path = cache_dir / "tag2idx.json"
tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

# Model Loading
if os.path.isfile('slot_Bi_LSTM_model.h5') is False:
    print('Cant detect model, please make sure model been downloaded.')
else:
    model = load_model('slot_Bi_LSTM_model.h5')

# Do testing data loading/prediction with model 
def main(argv):
    test_data_path = argv[1]
    predictions_labels_path = argv[2]

    with open(test_data_path) as data_f:
        data = json.load(data_f)

    tokens = [info['tokens'] for info in data]
    sequences = [vocab.encode(_tokens) for _tokens in tokens]
    test_shape = [len(info['tokens']) for info in data]
    test_id = [info['id'] for info in data]
    sequences = pad_sequences(sequences, padding='post', maxlen =max_len,
                                value =0, dtype ='int32')
    sequences = sequences.reshape(-1,)

# Predict test slot
    predictions = model.predict(sequences)
    predictions = np.argmax(predictions, axis=-1)
    predictions = predictions.reshape(len(test_id),-1)

    slot_predictions = {}
    for n in range(len(predictions)):
        slot_predictions[test_id[n]] = predictions[n][:test_shape[n]]     

    idx2slot = {idx:intent for intent,idx in tag2idx.items()}

    for n in range(len(slot_predictions)):
        slot_predictions[test_id[n]] = [idx2slot[idx] for idx in slot_predictions[test_id[n]]]

# Output prediction
    with open(predictions_labels_path, 'w', encoding='UTF8') as f:
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

if __name__ == "__main__":
    main(sys.argv)