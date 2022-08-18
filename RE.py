import torch
from models.BRNN_O import IntentIntegrateOnehot
from utils.data import SnortIntentBatchDataset, load_pkl
from torch.utils.data import DataLoader
from utils.utils import len_stats, pad_dataset, Logger
from ByteLevelTokenization.create_logic_mat_bias import create_mat_and_bias
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from fsa_to_tensor import dfa_to_tensor


def REclassifier(model, intent_dataloader, config=None, i2in=None, is_cuda=True):
    acc = 0

    model.eval()
    all_pred = []
    all_label = []
    all_out = []

    with torch.no_grad():
        for batch in intent_dataloader:

            x = batch['x']
            label = batch['i'].view(-1)
            lengths = batch['l']

            if torch.cuda.is_available() and is_cuda:
                x = x.cuda()
                lengths = lengths.cuda()
                label = label.cuda()

            out = model(x, lengths)

            acc += (out.argmax(1) == label).sum().item()

            all_pred += list(out.argmax(1).cpu().numpy())
            all_label += list(label.cpu().numpy())
            all_out.append(out.cpu().numpy())


    acc = acc / len(intent_dataloader.dataset)
    print('total acc: {}'.format(acc))

    if config.only_probe:
        confusion_mat = confusion_matrix(all_label, all_pred, labels=[i for i in range(config.label_size)])
        labels = [i2in[i] for i in range(config.label_size)]
        fig = plt.figure()
        fig.set_size_inches(8, 8)
        cmap = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=False)
        g = sns.heatmap(confusion_mat, annot=True,   cmap=cmap, linewidths=1,
                        linecolor='gray', xticklabels=labels, yticklabels=labels,)
        plt.show()

    p_micro = precision_score(all_label, all_pred, average='micro')
    r_micro = recall_score(all_label, all_pred, average='micro')
    f1_micro = f1_score(all_label, all_pred, average='micro')

    p_macro = precision_score(all_label, all_pred, average='macro')
    r_macro = recall_score(all_label, all_pred, average='macro')
    f1_macro = f1_score(all_label, all_pred, average='macro')

    # print('p_micro: {} | r_micro: {} | f1_micro: {}'.format(p_micro, r_micro, f1_micro))
    # print('p_macro: {} | r_macro: {} | f1_macro: {}'.format(p_macro, r_macro, f1_macro))

    print(f1_micro)

    return all_pred, np.concatenate(all_out)


def PredictByRE(args, params=None, dset=None,):
    logger = Logger()
    # if not dset:
    #     dset = load_classification_dataset(args.dataset)

    t2i, i2t, in2i, i2in = dset['t2i'], dset['i2t'], dset['in2i'], dset['i2in']
    query_train, intent_train = dset['query_train'], dset['intent_train']
    query_dev, intent_dev = dset['query_dev'], dset['intent_dev']
    query_test, intent_test = dset['query_test'], dset['intent_test']

    len_stats(query_train)
    len_stats(query_dev)
    len_stats(query_test)
    # extend the padding
    # add pad <pad> to the last of vocab
    i2t[len(i2t)] = '<pad>'
    t2i['<pad>'] = len(i2t) - 1

    train_query, train_query_inverse, train_lengths = pad_dataset(query_train, args, t2i['<pad>'])
    dev_query, dev_query_inverse, dev_lengths = pad_dataset(query_dev, args, t2i['<pad>'])
    test_query, test_query_inverse, test_lengths = pad_dataset(query_test, args, t2i['<pad>'])

    intent_data_train = SnortIntentBatchDataset(train_query, train_lengths, intent_train)
    intent_data_dev = SnortIntentBatchDataset(dev_query, dev_lengths, intent_dev)
    intent_data_test = SnortIntentBatchDataset(test_query, test_lengths, intent_test)

    intent_dataloader_train = DataLoader(intent_data_train, batch_size=args.bz)
    intent_dataloader_dev = DataLoader(intent_data_dev, batch_size=args.bz)
    intent_dataloader_test = DataLoader(intent_data_test, batch_size=args.bz)

    if params is None:
        automata_dicts = load_pkl(args.automata_path_forward)
        automata = automata_dicts['automata']
        language_tensor, state2idx, wildcard_mat, language = dfa_to_tensor(automata, t2i)
        complete_tensor = language_tensor + wildcard_mat
        mat, bias = create_mat_and_bias(automata, in2i=in2i, i2in=i2in,)
    else:
        complete_tensor = params['complete_tensor']
        mat, bias = params['mat'], params['bias']

    # for padding
    V, S1, S2 = complete_tensor.shape
    complete_tensor_extend = np.concatenate((complete_tensor, np.zeros((1, S1, S2))))
    print(complete_tensor_extend.shape)
    model = IntentIntegrateOnehot(complete_tensor_extend,
                                      config=args,
                                      mat=mat,
                                      bias=bias)

    if torch.cuda.is_available():
        model.cuda()
    # # TRAIN
    print('RE TRAIN ACC')
    all_pred_train, all_out_train = REclassifier(model, intent_dataloader_train,  config=args, i2in=i2in)
    # DEV
    print('RE DEV ACC')
    all_pred_dev, all_out_dev = REclassifier(model, intent_dataloader_dev, config=args,  i2in=i2in)
    # TEST
    print('RE TEST ACC')
    all_pred_test, all_out_test = REclassifier(model, intent_dataloader_test, config=args,i2in=i2in)

    return all_pred_train, all_pred_dev, all_pred_test, all_out_train, all_out_dev, all_out_test


def PredictByRE1(args, params=None, dset=None, gpu=0):
    logger = Logger()
    device = torch.device("cuda:{}".format(gpu))

    # if not dset:
    #     dset = load_classification_dataset(args.dataset)

    t2i, i2t, in2i, i2in = dset['t2i'], dset['i2t'], dset['in2i'], dset['i2in']
    query_train, intent_train = dset['query_train'], dset['intent_train']
    query_dev, intent_dev = dset['query_dev'], dset['intent_dev']
    query_test, intent_test = dset['query_test'], dset['intent_test']

    len_stats(query_train)
    len_stats(query_dev)
    len_stats(query_test)
    # extend the padding
    # add pad <pad> to the last of vocab
    i2t[len(i2t)] = '<pad>'
    t2i['<pad>'] = len(i2t) - 1

    train_query, train_query_inverse, train_lengths = pad_dataset(query_train, args, t2i['<pad>'])
    dev_query, dev_query_inverse, dev_lengths = pad_dataset(query_dev, args, t2i['<pad>'])
    test_query, test_query_inverse, test_lengths = pad_dataset(query_test, args, t2i['<pad>'])

    intent_data_train = SnortIntentBatchDataset(train_query, train_lengths, intent_train)
    intent_data_dev = SnortIntentBatchDataset(dev_query, dev_lengths, intent_dev)
    intent_data_test = SnortIntentBatchDataset(test_query, test_lengths, intent_test)

    intent_dataloader_train = DataLoader(intent_data_train, batch_size=args.bz)
    intent_dataloader_dev = DataLoader(intent_data_dev, batch_size=args.bz)
    intent_dataloader_test = DataLoader(intent_data_test, batch_size=args.bz)

    if params is None:
        automata_dicts = load_pkl(args.automata_path)
        automata = automata_dicts['automata']
        language_tensor, state2idx, wildcard_mat, language = dfa_to_tensor(automata, t2i)
        complete_tensor = language_tensor + wildcard_mat
        mat, bias = create_mat_and_bias(automata, in2i=in2i, i2in=i2in,)
    else:
        complete_tensor = params['complete_tensor']
        mat, bias = params['mat'], params['bias']

    # for padding
    V, S1, S2 = complete_tensor.shape
    complete_tensor_extend = np.concatenate((complete_tensor, np.zeros((1, S1, S2))))
    print(complete_tensor_extend.shape)
    model = IntentIntegrateOnehot(complete_tensor_extend,
                                  config=args,
                                  mat=mat,
                                  bias=bias,
                                  is_cuda=False)

    # TRAIN
    print('RE TRAIN ACC')
    all_pred_train, all_out_train = REclassifier(model, intent_dataloader_train,  config=args, i2in=i2in, is_cuda=False)
    # DEV
    print('RE DEV ACC')
    all_pred_dev, all_out_dev = REclassifier(model, intent_dataloader_dev, config=args,  i2in=i2in, is_cuda=False)
    # TEST
    print('RE TEST ACC')
    all_pred_test, all_out_test= REclassifier(model, intent_dataloader_test, config=args,i2in=i2in, is_cuda=False)

    return all_pred_train, all_pred_dev, all_pred_test, all_out_train, all_out_dev, all_out_test


