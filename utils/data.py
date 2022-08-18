from torch.utils.data import Dataset
import os
import numpy as np
import pickle
from collections import Counter
from utils import mkdir, evan_select_from_total_number

class SnortIntentDataset(Dataset):

    def __init__(self, query, intent):
        assert len(query) == len(intent)
        self.dataset = query
        self.intent = intent

    def __getitem__(self, idx):
        return {
            'x':self.dataset[idx],
            'i':self.intent[idx]
        }

    def __len__(self):
        return len(self.dataset)


class SnortIntentBatchDataset(Dataset):

    def __init__(self, query, lengths, intent, shots=None):
        assert len(query) == len(intent)
        if (not shots) or (shots > len(query)):
            self.dataset = query
            self.intent = intent
            self.lengths = lengths
        else:
            # idxs = np.random.choice(np.arange(len(query)), size=int(portion* len(query)), replace=False)
            idxs = evan_select_from_total_number(len(query), shots)
            self.dataset = list(np.array(query)[idxs])
            self.intent = list(np.array(intent)[idxs])
            self.lengths = list(np.array(lengths)[idxs])

    def __getitem__(self, idx):

        return {
            'x': np.array(self.dataset[idx]),
            'i': np.array(self.intent[idx]),
            'l': np.array(self.lengths[idx])
        }

    def __len__(self):
        return len(self.dataset)


class SnortIntentBatchDatasetBidirection(Dataset):

    def __init__(self, query, query_inverse, lengths, intent, shots=None):
        assert len(query) == len(intent)
        if (not shots) or (shots > len(query)):
            self.dataset = query
            self.dataset_inverse = query_inverse
            self.intent = intent
            self.lengths = lengths
        else:
            idxs = evan_select_from_total_number(len(query), shots)
            self.dataset = list(np.array(query)[idxs])
            self.dataset_inverse = list(np.array(query_inverse)[idxs])
            self.intent = list(np.array(intent)[idxs])
            self.lengths = list(np.array(lengths)[idxs])

    def __getitem__(self, idx):

        return {
            'x_forward': np.array(self.dataset[idx]),
            'x_backward': np.array(self.dataset_inverse[idx]),
            'i': np.array(self.intent[idx]),
            'l': np.array(self.lengths[idx])
        }

    def __len__(self):
        return len(self.dataset)

class SnortIntentBatchDatasetUtilizeUnlabel(Dataset):

    def __init__(self, query, query_inverse, lengths, intent_gold, intent_re,  re_out, shots=None):
        assert len(query) == len(intent_gold)
        assert len(query) == len(intent_re)
        self.dataset = query
        self.lengths = lengths
        self.re_out = re_out
        self.dataset_inverse = query_inverse

        if (shots == None) or (shots > len(query)):
            self.intent = intent_gold
        elif shots == 0:
            self.intent = intent_re
        else:
            idxs = evan_select_from_total_number(len(query), shots)
            new_intent = np.array(intent_re)
            selected = np.array(intent_gold)[idxs].reshape(-1)
            new_intent[idxs] = selected
            self.intent = list(new_intent)

    def __getitem__(self, idx):

        return {
            'x_forward': np.array(self.dataset[idx]),
            'x_backward': np.array(self.dataset_inverse[idx]),
            'i': np.array(self.intent[idx]),
            'l': np.array(self.lengths[idx])
        }

    def __len__(self):
        return len(self.dataset)


class MarryUpIntentBatchDataset(Dataset):

    def __init__(self, query, lengths, intent, re_out, shots=None):
        assert len(query) == len(intent)
        if (shots == None) or (shots > len(query)):
            self.dataset = query
            self.intent = intent
            self.lengths = lengths
            self.re_out = re_out
        else:
            idxs = evan_select_from_total_number(len(query), shots)
            self.dataset = list(np.array(query)[idxs])
            self.intent = list(np.array(intent)[idxs])
            self.lengths = list(np.array(lengths)[idxs])
            self.re_out = list(np.array(re_out)[idxs])

    def __getitem__(self, idx):

        return {
            'x': np.array(self.dataset[idx]),
            'i': np.array(self.intent[idx]),
            'l': np.array(self.lengths[idx]),
            're': np.array(self.re_out[idx])
        }

    def __len__(self):
        return len(self.dataset)

class MarryUpIntentBatchDatasetUtilizeUnlabel(Dataset):

    def __init__(self, query, lengths, intent_gold, intent_re,  re_out, shots=None):
        assert len(query) == len(intent_gold)
        assert len(query) == len(intent_re)
        self.dataset = query
        self.lengths = lengths
        self.re_out = re_out
        if (shots == None) or (shots > len(query)):
            self.intent = intent_gold
        elif shots == 0:
            self.intent = intent_re
        else:
            idxs = evan_select_from_total_number(len(query), shots)
            new_intent = np.array(intent_re)
            selected = np.array(intent_gold)[idxs].reshape(-1)
            new_intent[idxs] = selected
            self.intent = list(new_intent)

    def __getitem__(self, idx):

        return {
            'x': np.array(self.dataset[idx]),
            'i': np.array(self.intent[idx]),
            'l': np.array(self.lengths[idx]),
            're': np.array(self.re_out[idx])
        }

    def __len__(self):
        return len(self.dataset)

def load_ds(fname='../data/ATIS/atis.train.pkl'):
    print(os.listdir())
    with open(fname,'rb') as stream:
        ds, dicts = pickle.load(stream)
    print('Done  loading: ', fname)
    print('      samples: {:4d}'.format(len(ds['query'])))
    print('   vocab_size: {:4d}'.format(len(dicts['token_ids'])))
    print('   slot count: {:4d}'.format(len(dicts['slot_ids'])))
    print(' intent count: {:4d}'.format(len(dicts['intent_ids'])))
    return ds, dicts

def load_pkl(path):
    print(path)
    dicts = pickle.load(open(path, 'rb'))
    return dicts

def save_data(query, slots, intent, mode, DATA_DIR = '../data/ATIS'):
    pickle.dump((query, slots, intent), open(os.path.join(DATA_DIR, 'atis.{}.new.pkl'.format(mode)), 'wb'))

def create_vocabs(iterable, mode):
    assert mode in ['labels', 'texts']
    vocab = Counter()
    if mode == 'labels':
        vocab = vocab + Counter(list(iterable))
    else:
        for instance in iterable:
            vocab += Counter(instance)

    vocab_list = list(vocab.keys())
    i2v = {idx: vocab for idx, vocab in enumerate(vocab_list)}
    v2i = {vocab: idx for idx, vocab in enumerate(vocab_list)}

    return i2v, v2i

