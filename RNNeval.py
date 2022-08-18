import os
import torch
import torch.nn as nn
from src.models.FARNN import IntentIntegrateSaperate_B, IntentIntegrateSaperateBidirection_B, \
    FSARNNIntegrateEmptyStateSaperateGRU
from src.models.FARNN_O import IntentIntegrateOnehot
from src.models.Baseline import IntentMarryUp
from src.data import ATISIntentBatchDataset, ATISIntentBatchDatasetBidirection, ATISIntentBatchDatasetUtilizeUnlabel, \
    MarryUpIntentBatchDataset, \
    load_glove_embed, load_pkl, MarryUpIntentBatchDatasetUtilizeUnlabel
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils.utils import len_stats, pad_dataset, mkdir, create_datetime_str, Logger, relu_normalized_NLLLoss
from src.rules.create_logic_mat_bias import create_mat_and_bias, create_mat_and_bias_with_empty_ATIS, \
    create_mat_and_bias_with_empty_TREC, create_mat_and_bias_with_empty_SMS
import numpy as np
from src.RE import PredictByRE
from src.rules.fsa_to_tensor import dfa_to_tensor
import pickle
from src.val import val, val_marry
from copy import deepcopy
from load_dataset import load_classification_dataset
import argparse
from src.models.FARNN_O import IntentIntegrateOnehot

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=0, help="cuda:0,1,2,3,4,5")
parser.add_argument('--dataset', type=str, default='policy', help="dataset name")
parser.add_argument('--dataset_spilt', type=int, default=1, help="rate of using labeled data, [0.01, 0.1, 1]")
parser.add_argument('--test_spilt', type=float, default=0.2, help="spilt rate of test set")
parser.add_argument('--val_spilt', type=float, default=0.1, help="spilt rate of validation set")
parser.add_argument('--lr', type=float, default=0.0001, help="learning rate of optimizer")
parser.add_argument('--max_state', type=int, default=3, help="max state of each FSARNN")
parser.add_argument('--bz', type=int, default=500, help="batch size")
parser.add_argument('--epoch', type=int, default=100, help="max state of each FSARNN")
parser.add_argument('--seq_max_len', type=int, default=50, help="max length of sequence")
parser.add_argument('--early_stop', type=int, default=5,
                    help="number of epochs that apply early stopping if dev metric not improve")
parser.add_argument('--run', type=str, default="integrate", help="run folder name to save model")
parser.add_argument('--seed', type=str, default='0', help="random seed")
parser.add_argument('--model_id', type=str, default='none', help="model id, identifying model")
parser.add_argument('--activation', type=str, default='relu',
                    help='nonlinear of model, [tanh, sigmoid, relu, relu6]')
parser.add_argument('--clamp_hidden', type=int, default=0,
                    help='if constrain the hidden layer at each time stamp to between 0, 1')

parser.add_argument('--train_fsa', type=int, default=1, help="if we train the fsa parameters")
parser.add_argument('--train_wildcard', type=int, default=1, help='If we train the wildcard matrix')
parser.add_argument('--train_linear', type=int, default=1, help="if we train the linear parameters")
parser.add_argument('--train_V_embed', type=int, default=1, help="if we train the V_embed parameters")
parser.add_argument('--train_word_embed', type=int, default=1, help="if we train the word embed parameters")
parser.add_argument('--train_beta', type=int, default=1, help="if we train the vector beta")
parser.add_argument('--clip_neg', type=int, default=0, help="if we use relu after each hidden update")
parser.add_argument('--sigmoid_exponent', type=int, default=5, help="exponent in TENSORFSAGRU sigmoid")
parser.add_argument('--beta', type=float, default=1.0,
                    help="in [0,1] interval, 1 means use regex, 0 means only use embedding")
parser.add_argument('--farnn', type=int, default=1, help="0 if use farnn, 1 if use fagru")
parser.add_argument('--bias_init', type=float, default=7, help="bias_init for gru")
parser.add_argument('--reg_type', type=str, default='nuc', help="nuclear norm or fro norm")
parser.add_argument('--wfa_type', type=str, default='forward', help='forward or viterbi')
parser.add_argument('--random_embed', type=int, default=1, help='random embedding or not')
parser.add_argument('--clamp_score', type=int, default=0, help='if we clamp the score in [0, 1]')
parser.add_argument('--rnn_hidden_dim', type=int, default=200, help='rnn hidden dim')
parser.add_argument('--model_type', type=str, default='Onehot', help='baseline MarryUp or FSARNN')
parser.add_argument('--optimizer', type=str, default='ADAM', help='optimizer SGD or ADAM')
parser.add_argument('--additional_nonlinear', type=str, default='none',
                    help='additional nonlinear after DxR, should be in [none, relu, sigmoid, tanh]')
parser.add_argument('--additional_state', type=int, default=0, help='additional reserved state for generalization')

parser.add_argument('--regularization', type=str, default='V_embed_weighted',
                    help="[V_embed, V_embed_weighted, D1, D2, None]")
parser.add_argument('--only_probe', type=int, default=0, help='if we only prob and not train')
parser.add_argument('--rnn', type=str, default='RNN', help='rnn type only for MarryUp Baseline')
parser.add_argument('--re_tag_dim', type=int, default=20, help='re tag dim only for MarryUp Baseline')
parser.add_argument('--marryup_type', type=str, default='none',
                    help='only for marry up baseline, should be in [input, output, all, none]')
parser.add_argument('--train_portion', type=float, default=1.0, help='training portion, in 0.01, 0.1, 1.0 ')
parser.add_argument('--random', type=int, default=0, help='if use random initialzation')

parser.add_argument('--l1', type=float, default=0, help="kd alpha")
parser.add_argument('--l2', type=float, default=0., help="pr constant")

parser.add_argument('--bidirection', type=int, default=0, help='if the model is bidirection')
parser.add_argument('--xavier', type=int, default=1, help='if the FSAGRU model initialized using xavier')
parser.add_argument('--normalize_automata', type=str, default='l2',
                    help='how to normalize the automata, none, l1, l2, avg')
parser.add_argument('--random_noise', type=float, default=0.001,
                    help='random noise used when adding additional states')
parser.add_argument('--loss_type', type=str, default='CrossEntropy', help='CrossEntropy, NormalizeNLL')
parser.add_argument('--use_unlabel', type=int, default=0, help='use unlabel or not')
parser.add_argument('--embed_dim', type=int, default=100, help='embed dim')

parser.add_argument('--automata_path_forward', type=str, default='none', help="automata path")
parser.add_argument('--automata_path_backward', type=str, default='none', help="automata path")
parser.add_argument('--model_name', type=str, default='chat-0.01', help="automata path")
parser.add_argument('--cuda_num', type=int, default=0, help="cuda to test same to train")

args = parser.parse_args()

cuda_num = args.cuda_num
torch.cuda.set_device(cuda_num)

dset = load_classification_dataset(args)
t2i, i2t, in2i, i2in = dset['t2i'], dset['i2t'], dset['in2i'], dset['i2in']

query_train, intent_train = dset['query_train'], dset['intent_train']
query_dev, intent_dev = dset['query_dev'], dset['intent_dev']
query_test, intent_test = dset['query_test'], dset['intent_test']

len_stats(query_train)
len_stats(query_dev)
len_stats(query_test)
i2t[len(i2t)] = '<pad>'
t2i['<pad>'] = len(i2t) - 1

train_query, train_query_inverse, train_lengths = pad_dataset(query_train, args, t2i['<pad>'])
dev_query, dev_query_inverse, dev_lengths = pad_dataset(query_dev, args, t2i['<pad>'])
test_query, test_query_inverse, test_lengths = pad_dataset(query_test, args, t2i['<pad>'])

shots = int(len(train_query) * args.train_portion)
assert args.train_portion == 1.0
intent_data_train = ATISIntentBatchDataset(train_query, train_lengths, intent_train, shots)
intent_data_dev = ATISIntentBatchDataset(dev_query, dev_lengths, intent_dev, shots)
intent_data_test = ATISIntentBatchDataset(test_query, test_lengths, intent_test)


intent_dataloader_train = DataLoader(intent_data_train, batch_size=500)
intent_dataloader_dev = DataLoader(intent_data_dev, batch_size=500)
intent_dataloader_test = DataLoader(intent_data_test, batch_size=500)

print(os.path.join('model', args.model_name, args.model_name+'.m'))
model = torch.load(os.path.join('model', args.model_name, args.model_name+'.m'))
if torch.cuda.is_available():
    model = model.cuda(cuda_num)

model.eval()

avg_loss = 0
acc = 0
pbar_train = intent_dataloader_train
train_x = []
train_label = []
train_score = []

for batch in pbar_train:
    x = batch['x']
    label = batch['i'].view(-1)
    lengths = batch['l']
    train_x.append(x.numpy())
    train_label.append(label.numpy())
    if torch.cuda.is_available():
        x = x.cuda(cuda_num)
        lengths = lengths.cuda(cuda_num)
        label = label.cuda(cuda_num)

    scores = model(x, lengths)
    train_score.append(scores.cpu().detach().numpy())
if os.path.exists(os.path.join('Distillation', 'softLabel', args.model_name)) == False:
    os.makedirs(os.path.join('Distillation', 'softLabel', args.model_name))
x_arr = train_x[0]
for i in range(1, len(train_x)):
    x_arr = np.append(x_arr, train_x[i], axis=0)

np.save(file=os.path.join('Distillation', 'softLabel', args.model_name,'train_x.npy'), arr=x_arr)
print(x_arr.shape)

label_arr = train_label[0]
for i in range(1, len(train_label)):
    label_arr = np.append(label_arr, train_label[i], axis=0)

np.save(file=os.path.join('Distillation', 'softLabel', args.model_name,'train_label.npy'), arr=label_arr)
print(label_arr.shape)

score_arr = train_score[0]
for i in range(1, len(train_score)):
    score_arr = np.append(score_arr, train_score[i], axis=0)

np.save(file=os.path.join('Distillation', 'softLabel', args.model_name,'train_score.npy'), arr=score_arr)
print(score_arr.shape)


pbar_test = intent_dataloader_test
test_x = []
test_label = []
test_score = []
for batch in pbar_test:
    x = batch['x']
    label = batch['i'].view(-1)
    lengths = batch['l']
    test_x.append(x.numpy())
    test_label.append(label.numpy())
    if torch.cuda.is_available():
        x = x.cuda(cuda_num)
        lengths = lengths.cuda(cuda_num)
        label = label.cuda(cuda_num)
    scores = model(x, lengths)
    test_score.append(scores.cpu().detach().numpy())

x_arr = test_x[0]
for i in range(1, len(test_x)):
    x_arr = np.append(x_arr, test_x[i], axis=0)

np.save(file=os.path.join('Distillation', 'softLabel', args.model_name,'test_x.npy'), arr=x_arr)
print(x_arr.shape)

label_arr = test_label[0]
for i in range(1, len(test_label)):
    label_arr = np.append(label_arr, test_label[i], axis=0)

np.save(file=os.path.join('Distillation', 'softLabel', args.model_name,'test_label.npy'), arr=label_arr)
print(label_arr.shape)

score_arr = test_score[0]
for i in range(1, len(test_score)):
    score_arr = np.append(score_arr, test_score[i], axis=0)

np.save(file=os.path.join('Distillation', 'softLabel', args.model_name,'test_score.npy'), arr=score_arr)
print(score_arr.shape)
print(label_arr[0:10])
print(score_arr[0:10])





