import os
import argparse
import pandas as pd
from collections import Counter
from typing import List
import pickle

import shutil


def HandleBytes(b: str):
    b = b[2:]
    output = []
    byte_ = '0x'
    for n, c in enumerate(b):
        byte_ += c
        if n != 0 and (n + 1) % 2 == 0:
            output.append(byte_)
            byte_ = '0x'
    return output


def get_word_to_index(texts: List[List[str]]):
    vocab = Counter()
    for text in texts:
        vocab += Counter(text)
    vocabList = list(vocab.keys())
    l = len(vocabList)
    indexToWord = {idx: vocab for idx, vocab in enumerate(vocabList)}
    indexToWord[l] = 'BOS'
    indexToWord[l + 1] = 'EOS'
    wordToIndex = {vocab: idx for idx, vocab in enumerate(vocabList)}
    wordToIndex['BOS'] = l
    wordToIndex['EOS'] = l + 1

    return indexToWord, wordToIndex


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


def load_snort_dataset(args):
    file_path = os.path.dirname(__file__) + "/data/snort/{}/{}.csv".format(args.dataset, args.dataset)
    df = pd.read_csv(file_path, header=None).reset_index(drop=True).rename({'0': 'class', '1': 'text'}, axis='columns')
    df.columns = ["class", "text"]
    # print(df.head(3))
    df = df.sample(frac=1.0, random_state=2)  # shaffle
    test_idx = int(round(args.test_spilt * df.shape[0]))
    val_idx = int(round(args.val_spilt * df.shape[0]))
    df_test, df_val, df_train = df.iloc[:test_idx], df.iloc[test_idx:test_idx + val_idx], df.iloc[test_idx + val_idx:]

    df_test['mode'] = 'test'
    df_val['mode'] = 'valid'
    df_train['mode'] = 'train'

    df = pd.concat([df_test, df_train, df_val], ignore_index=True)
    indexToWord, wordToIndex = get_word_to_index(map(lambda text: HandleBytes(text), list(df['text'])))

    # print(list(df['text']))
    print(wordToIndex)
    return {
        'data': df,
        'indexToWord': indexToWord,
        'wordToIndex': wordToIndex
    }


# def load_snort_dataset(args):
#     file_path = os.path.dirname(__file__) + "/data/snort/{}/{}.csv".format(args.dataset_name, args.dataset_name)
#     df = pd.read_csv(file_path).reset_index(drop=True).rename({'v1': 'class', 'v2': 'text'}, axis='columns')
#     df = df.sample(frac=1.0)  # shaffle
#     test_idx = int(round(args.test_spilt * df.shape[0]))
#     val_idx = int(round(args.val_spilt * df.shape[0]))
#     df_test, df_val, df_train = df.iloc[:test_idx], df.iloc[test_idx:test_idx + val_idx], df.iloc[test_idx + val_idx:]
#
#     df_test['mode'] = 'test'
#     df_val['mode'] = 'valid'
#     df_train['mode'] = 'train'
#
#     df = pd.concat([df_test, df_train, df_val], ignore_index=True)
#
#     indexToWord, wordToIndex = get_word_to_index(map(lambda text: HandleBytes(text), list(df['text'])))
#
#     return {
#         'data': df,
#         'indexToWord': indexToWord,
#         'wordToIndex': wordToIndex
#     }

def create_classification_dataset(args):
    res = load_snort_dataset(args)

    print('CREATING VOCAB FILES')
    data = res['data']
    labels = list(data['class'])
    texts = list(data['text'])
    # print(texts)
    # texts = [['BOS'] + i.strip().split() + ['EOS'] for i in texts]
    i2in, in2i = create_vocabs(labels, 'labels')
    in2i[1] = 1
    in2i[0] = 0
    i2in[1] = 1
    i2in[0] = 0
    i2t, t2i = res['indexToWord'], res['wordToIndex']

    print('TRANSFORMING TO INDEX')
    data = data.groupby('mode')
    train, dev, test = data.get_group('train'), data.get_group('valid'), data.get_group('test')

    def to_query_intent(dataset):
        labels = dataset['class']
        # print(dataset['text'])
        texts = [HandleBytes(text) for text in dataset['text']]
        # print(texts)
        texts = [['BOS'] + i + ['EOS'] for i in texts]
        intent = [in2i[i] for i in labels]
        query = [[t2i[j] for j in i] for i in texts]
        # print(query)
        return intent, query

    intent_train, query_train = to_query_intent(train)

    dataset_spilt_idx = int(round(args.dataset_spilt * train.shape[0]))
    intent_train = intent_train[:dataset_spilt_idx]
    query_train = query_train[:dataset_spilt_idx]

    intent_dev, query_dev = to_query_intent(dev)
    intent_test, query_test = to_query_intent(test)

    print('SAVING DATASET')
    dataset = {
        't2i': t2i, 'i2t': i2t, 'in2i': in2i, 'i2in': i2in,
        'query_train': query_train, 'intent_train': intent_train,
        'query_dev': query_dev, 'intent_dev': intent_dev,
        'query_test': query_test, 'intent_test': intent_test,
    }
    print(in2i)
    pickle.dump(dataset, open(
        os.path.dirname(__file__) + '/data/snort/{}/dataset-{}.pkl'.format(args.dataset, args.dataset_spilt), 'wb'))
    # print(dataset['in2i'])


def load_classification_dataset(args):
    file_path = os.path.dirname(__file__) + '/data/snort/{}/dataset-{}.pkl'.format(args.dataset, args.dataset_spilt)
    if not os.path.exists(file_path):
        create_classification_dataset(args)

    dataset = pickle.load(open(file_path, 'rb'))
    return dataset


if __name__ == '__main__':

    snort_dataset = ['chat',
                     'ftp',
                     'games',
                     'misc',
                     'netbios',
                     'p2p',
                     'policy',
                     'scan',
                     'telnet',
                     'tftp',
                     'web_client', ]

    # snort_dataset = ['policy']

    for snort in snort_dataset:

        scoure_csv = os.path.dirname(__file__) + "/LabeledData/{}.csv".format(snort)
        dist = os.path.dirname(__file__) + "/data/snort/{}/{}.csv".format(snort, snort)
        shutil.copy(scoure_csv, dist)
        for spilt in [0.01, 0.1, 1]:
            parser = argparse.ArgumentParser()
            parser.add_argument('--dataset', type=str, default=snort, help="dataset name")
            parser.add_argument('--test_spilt', type=float, default=0.2, help="spilt rate of test set")
            parser.add_argument('--val_spilt', type=float, default=0.1, help="spilt rate of validation set")
            parser.add_argument('--dataset_spilt', type=float, default=spilt, help="rate of using labeled data")

            args = parser.parse_args()
            # assert args.dataset_name in ['ATIS', 'TREC', 'SMS']

            print(create_classification_dataset(args))
