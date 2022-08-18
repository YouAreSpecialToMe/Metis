import numpy as np
import random
import torch
import os
import datetime, time


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def len_stats(query):
    max_len = 0
    avg_len = 0
    for q in query:
        max_len = max(len(q), max_len)
        avg_len += len(q)
    avg_len /= len(query)

    print("max_len: {}, avg_len: {}".format(max_len, avg_len))


def pad_dataset(query, config, pad_idx):
    lengths = []
    new_query = []
    new_query_inverse = []
    for q in query:
        length = len(q)
        q_inverse = q[::-1]

        if length > config.seq_max_len:
            q = q[: config.seq_max_len]
            q_inverse = q_inverse[: config.seq_max_len]
            length = config.seq_max_len
        else:
            remain = config.seq_max_len - length
            remain_arr = np.repeat(pad_idx, remain)
            q = np.concatenate((q, remain_arr))
            q_inverse = np.concatenate((q_inverse, remain_arr))
            assert len(q) == config.seq_max_len

        new_query.append(q)
        new_query_inverse.append(q_inverse)
        lengths.append(length)

    return new_query, new_query_inverse, lengths


def pad_dataset_1(query, seq_max_len, pad_idx):
    lengths = []
    new_query = []
    new_query_inverse = []
    for q in query:
        length = len(q)

        if length <= 0:
            continue

        q_inverse = q[::-1]

        if length > seq_max_len:
            q = q[: seq_max_len]
            q_inverse = q_inverse[: seq_max_len]
            length = seq_max_len
        else:
            remain = seq_max_len - length
            remain_arr = np.repeat(pad_idx, remain)
            q = np.concatenate((q, remain_arr))
            q_inverse = np.concatenate((q_inverse, remain_arr))
            assert len(q) == seq_max_len

        new_query.append(q)
        new_query_inverse.append(q_inverse)
        lengths.append(length)

    return new_query, new_query_inverse, lengths


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def create_datetime_str():
    datetime_dt = datetime.datetime.today()
    datetime_str = datetime_dt.strftime("%m%d%H%M%S")
    datetime_str = datetime_str + '-' + str(time.time())
    return datetime_str


class Args():
    def __init__(self, data):
        self.data = data
        for k, v in data.items():
            setattr(self, k, v)


class Logger():
    def __init__(self):
        self.record = []  # recored strings

    def add(self, string):
        assert type(string) == str
        self.record.append(string + ' \n')

    def save(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            f.writelines(self.record)

def relu_normalized_NLLLoss(input, target):
    relu = torch.nn.ReLU()
    loss = torch.nn.NLLLoss()
    input = relu(input)
    input += 1e-4  # add small positive offset to prevent 0
    input = input / torch.sum(input)
    input = torch.log(input)
    loss_val = loss(input, target)
    return loss_val


def even_select_from_portion(L, portion):
    final_nums = int(L * portion)
    interval = 1 / portion
    idxs = [int(i * interval) for i in range(final_nums)]
    return np.array(idxs)


def evan_select_from_total_number(L, N):
    assert L >= N
    if N > 0:
        portion = N / L
        interval = 1 / portion
        idxs = [int(i * interval) for i in range(N)]
    else:
        idxs = []
    return np.array(idxs)
