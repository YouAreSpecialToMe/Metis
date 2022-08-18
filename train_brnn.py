import os
import torch
from models.BRNN_O import IntentIntegrateOnehot
from models.Baseline import IntentMarryUp
from utils.data import SnortIntentBatchDataset, MarryUpIntentBatchDataset,  load_pkl, MarryUpIntentBatchDatasetUtilizeUnlabel
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import len_stats, pad_dataset, mkdir, Logger, relu_normalized_NLLLoss
from ByteLevelTokenization.create_logic_mat_bias import create_mat_and_bias
import numpy as np
from RE import PredictByRE
from ByteLevelTokenization.fsa_to_tensor import dfa_to_tensor
import pickle
from val import val, val_marry
from copy import deepcopy
from ByteLevelTokenization.load_dataset import load_classification_dataset
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt


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
        g = sns.heatmap(confusion_mat, annot=True, cmap=cmap, linewidths=1,
                        linecolor='gray', xticklabels=labels, yticklabels=labels, )
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


def PredictByRE(args, params=None, dset=None, ):
    logger = Logger()
    if not dset:
        dset = load_classification_dataset(args)

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
        automata = automata_dicts
        language_tensor, state2idx, wildcard_mat, language = dfa_to_tensor(automata, t2i)
        complete_tensor = language_tensor + wildcard_mat
        mat, bias = create_mat_and_bias(automata, in2i=in2i, i2in=i2in, )
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
    all_pred_train, all_out_train = REclassifier(model, intent_dataloader_train, config=args, i2in=i2in)
    # DEV
    print('RE DEV ACC')
    all_pred_dev, all_out_dev = REclassifier(model, intent_dataloader_dev, config=args, i2in=i2in)
    # TEST
    print('RE TEST ACC')
    all_pred_test, all_out_test = REclassifier(model, intent_dataloader_test, config=args, i2in=i2in)

    return all_pred_train, all_pred_dev, all_pred_test, all_out_train, all_out_dev, all_out_test


def get_average(M, normalize_type):
    """
    :param M:
    :param normalize_type:
    :return:
    Get the averaged norm
    """
    assert normalize_type in ['l1', 'l2']
    Dims = M.shape
    eles = 1
    for D in Dims:
        eles *= D
    if normalize_type == 'l1':
        temp = np.linalg.norm(M, 1)
    elif normalize_type == 'l2':
        temp = np.linalg.norm(M, 2)

    return temp / eles


def get_init_params(config, in2i, i2in, t2i, automata_path):
    dset = config.dataset

    # pretrained_embed = torch.load(os.path.dirname(
    #     __file__) + '/model/integrate/FARNN-D0.9976-T0.9958-DI0.5006-TI0.4952-0707163801-1657183081.3714988-telnet-0:1:2:3.model')[
    #     'fsa_rnn.embedding.weight'].cpu().numpy()
    if config.random_embed:
        pretrained_embed = np.random.random((len(t2i) - 1, config.embed_dim))

    # print(t2i)

    automata_dicts = load_pkl(automata_path)
    automata = automata_dicts['automata']

    V_embed, D1, D2 = automata_dicts['V'], automata_dicts['D1'], automata_dicts['D2']
    wildcard_mat, language = automata_dicts['wildcard_mat'], automata_dicts['language']

    # print(V_embed.shape)
    n_vocab, rank = V_embed.shape
    n_state, _ = D1.shape
    print("DFA states: {}".format(n_state))
    _, embed_dim = pretrained_embed.shape

    mat, bias = create_mat_and_bias(automata, in2i=in2i, i2in=i2in, model_type=config.model_type)

    # for padding
    pretrain_embed_extend = pretrained_embed
    if config.random_embed:
        pretrain_embed_extend = np.append(pretrained_embed, np.zeros((1, config.embed_dim), dtype=np.float), axis=0)
    V_embed_extend = np.append(V_embed, np.zeros((1, rank), dtype=np.float), axis=0)

    # creating language mask for regularization
    n_vocab_extend, _ = V_embed_extend.shape
    language_mask = torch.ones(n_vocab_extend)
    language_mask[[t2i[i] for i in language]] = 0

    # for V_embed_weighted mask and extend the wildcard mat to the right dimension
    S, _ = wildcard_mat.shape
    wildcard_mat_origin_extend = np.zeros((S + config.additional_state, S + config.additional_state))
    wildcard_mat_origin_extend[:S, :S] = wildcard_mat
    wildcard_mat_origin_extend = torch.from_numpy(wildcard_mat_origin_extend).float()
    if torch.cuda.is_available():
        language_mask = language_mask.cuda()
        wildcard_mat_origin_extend = wildcard_mat_origin_extend.cuda()

    if config.normalize_automata != 'none':
        D1_avg = get_average(D1, config.normalize_automata)
        D2_avg = get_average(D2, config.normalize_automata)
        V_embed_extend_avg = get_average(V_embed_extend, config.normalize_automata)
        factor = np.float_power(D1_avg * D2_avg * V_embed_extend_avg, 1 / 3)
        print(factor)
        print(D1_avg)
        print(D2_avg)
        print(V_embed_extend_avg)

        D1 = D1 * (factor / D1_avg)
        D2 = D2 * (factor / D2_avg)
        V_embed_extend = V_embed_extend * (factor / V_embed_extend_avg)

    # print(V_embed_extend.shape)

    return V_embed_extend, pretrain_embed_extend, mat, bias, D1, D2, language_mask, language, wildcard_mat, wildcard_mat_origin_extend


def save_args_and_results(args, results, loggers):
    print('Saving Args and Results')
    # mkdir(os.path.dirname(__file__) + '/model/{}-{}'.format(args['dataset'], args['dataset_spilt']))
    # datetime_str = create_datetime_str()
    if args['model_type'] == 'Onehot':
        file_save_path = os.path.dirname(__file__) + "/model/{}-{}/{}-{}.res".format(
            args['dataset'], args['dataset_spilt'], args['dataset'], args['dataset_spilt']
        )
        print('Saving Args and Results at: {}'.format(file_save_path))
        pickle.dump({
            'args': args,
            'res': results,
            'loggers': loggers
        }, open(file_save_path, 'wb'))
    elif args['model_type'] == 'MarryUp':
        dir_path = os.path.dirname(__file__) + "/model/{}/".format(args['rnn'])
        if not os.path.exists(dir_path):
            mkdir(dir_path)
        snort_dir = dir_path + '{}-{}/'.format(args['dataset'], args['dataset_spilt'])
        if not os.path.exists(snort_dir):
            mkdir(snort_dir)
        file_save_path = os.path.dirname(__file__) + "/model/{}/{}-{}/{}-{}.res".format(
            args['rnn'], args['dataset'], args['dataset_spilt'], args['dataset'], args['dataset_spilt']
        )
        print('Saving Args and Results at: {}'.format(file_save_path))
        pickle.dump({
            'args': args,
            'res': results,
            'loggers': loggers
        }, open(file_save_path, 'wb'))


def train_onehot(args, paths):
    logger = Logger()

    torch.cuda.set_device(args.cuda)
    dset = load_classification_dataset(args)
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

    shots = int(len(train_query) * args.train_portion)
    assert args.train_portion == 1.0
    # We currently not support unlabel and low-resource for onehot
    intent_data_train = SnortIntentBatchDataset(train_query, train_lengths, intent_train, shots)
    intent_data_dev = SnortIntentBatchDataset(dev_query, dev_lengths, intent_dev, shots)
    intent_data_test = SnortIntentBatchDataset(test_query, test_lengths, intent_test)

    intent_dataloader_train = DataLoader(intent_data_train, batch_size=args.bz)
    intent_dataloader_dev = DataLoader(intent_data_dev, batch_size=args.bz)
    intent_dataloader_test = DataLoader(intent_data_test, batch_size=args.bz)

    automata_dicts = load_pkl(paths[0])

    if 'automata' not in automata_dicts:
        automata = automata_dicts
    else:
        automata = automata_dicts['automata']

    language_tensor, state2idx, wildcard_mat, language = dfa_to_tensor(automata, t2i)
    complete_tensor = language_tensor + wildcard_mat

    assert args.additional_state == 0

    mat, bias = create_mat_and_bias(automata, in2i=in2i, i2in=i2in, )

    # for padding
    V, S1, S2 = complete_tensor.shape
    complete_tensor_extend = np.concatenate((complete_tensor, np.zeros((1, S1, S2))))
    print(complete_tensor_extend.shape)
    model = IntentIntegrateOnehot(complete_tensor_extend,
                                  config=args,
                                  mat=mat,
                                  bias=bias)

    mode = 'onehot'
    if args.loss_type == 'CrossEntropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss_type == 'NormalizeNLL':
        criterion = relu_normalized_NLLLoss
    else:
        print("Wrong loss function")

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0)
    if args.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)

    if torch.cuda.is_available():
        model = model.cuda()

    acc_train_init, avg_loss_train_init, p, r = val(model, intent_dataloader_train, epoch=0, mode='TRAIN', config=args,
                                                    i2in=i2in, criterion=criterion)
    # DEV
    acc_dev_init, avg_loss_dev_init, p, r = val(model, intent_dataloader_dev, epoch=0, mode='DEV', config=args,
                                                i2in=i2in, criterion=criterion)
    # TEST
    acc_test_init, avg_loss_test_init, p, r = val(model, intent_dataloader_test, epoch=0, mode='TEST', config=args,
                                                  i2in=i2in, criterion=criterion)

    # pickle.dump(model, open("/home/dgl/zzx/PythonProject/RE2RNN/modelEval/{}.pkl".format(args.dataset), "wb"))

    best_dev_acc = acc_dev_init
    counter = 0
    best_dev_test_acc = acc_test_init

    for epoch in range(1, args.epoch + 1):
        avg_loss = 0
        acc = 0
        pbar_train = tqdm(intent_dataloader_train)
        pbar_train.set_description("TRAIN EPOCH {}".format(epoch))

        model.train()
        for batch in pbar_train:

            optimizer.zero_grad()

            x = batch['x']
            label = batch['i'].view(-1)
            lengths = batch['l']

            if torch.cuda.is_available():
                x = x.cuda()
                lengths = lengths.cuda()
                label = label.cuda()

            scores = model(x, lengths)

            loss_cross_entropy = criterion(scores, label)
            loss = loss_cross_entropy

            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

            acc += (scores.argmax(1) == label).sum().item()  # [0, 1] --> 1, [0, 0] --> 0

            pbar_train.set_postfix_str("{} - total right: {}, total loss: {}".format('TRAIN', acc, loss))

        acc = acc / len(intent_data_train)
        avg_loss = avg_loss / len(intent_data_train)

        print("{} Epoch: {} | ACC: {}, LOSS: {}".format('TRAIN', epoch, acc, avg_loss))
        logger.add("{} Epoch: {} | ACC: {}, LOSS: {}".format('TRAIN', epoch, acc, avg_loss))

        # DEV
        acc_dev, avg_loss_dev, p, r = val(model, intent_dataloader_dev, epoch, 'DEV', logger, config=args,
                                          criterion=criterion)
        # TEST
        acc_test, avg_loss_test, p, r = val(model, intent_dataloader_test, epoch, 'TEST', logger, config=args,
                                            criterion=criterion)

        counter += 1  # counter for early stopping

        if (acc_dev is None) or (acc_dev > best_dev_acc):
            counter = 0
            best_dev_acc = acc_dev
            best_dev_test_acc = acc_test

        if counter > args.early_stop:
            break

    # Save the model
    # datetime_str = create_datetime_str()
    model_save_path = os.path.dirname(__file__) + "/model/{}-{}/{}-{}".format(
        args.dataset, args.dataset_spilt, args.dataset, args.dataset_spilt
    )
    model_save_dir = os.path.dirname(__file__) + "/model/{}-{}/".format(args.dataset, args.dataset_spilt)
    if not os.path.exists(model_save_dir):
        mkdir(model_save_dir)
    print("SAVING MODEL {} .....".format(model_save_path))
    torch.save(model.state_dict(), model_save_path + '.model')
    torch.save(model, model_save_path + '.m')

    return acc_dev_init, acc_test_init, best_dev_acc, best_dev_test_acc, logger.record

def train_marry_up(args):
    torch.cuda.set_device(args.cuda)
    assert args.additional_state == 0
    if args.model_type == 'KnowledgeDistill':
        assert args.marryup_type == 'none'
    if args.model_type == 'PR':
        assert args.marryup_type == 'none'

    all_pred_train, all_pred_dev, all_pred_test, all_out_train, all_out_dev, all_out_test = PredictByRE(args)

    logger = Logger()
    # config = Config_MarryUp(args)

    dset = load_classification_dataset(args)
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

    train_query, _, train_lengths = pad_dataset(query_train, args, t2i['<pad>'])
    dev_query, _, dev_lengths = pad_dataset(query_dev, args, t2i['<pad>'])
    test_query, _, test_lengths = pad_dataset(query_test, args, t2i['<pad>'])

    shots = int(len(train_query) * args.train_portion)
    if args.use_unlabel:
        intent_data_train = MarryUpIntentBatchDatasetUtilizeUnlabel(train_query, train_lengths, intent_train,
                                                                    all_pred_train, all_out_train, shots)
    elif args.train_portion == 0:
        # special case when train portion==0 and do not use unlabel data, should have no data
        intent_data_train = None
    else:
        intent_data_train = MarryUpIntentBatchDataset(train_query, train_lengths, intent_train, all_out_train, shots)

    # should have no/few dev data in low-resource setting
    if args.train_portion == 0:
        intent_data_dev = None
    elif args.train_portion <= 0.01:
        intent_data_dev = MarryUpIntentBatchDataset(dev_query, dev_lengths, intent_dev, all_out_dev, shots)
    else:
        intent_data_dev = MarryUpIntentBatchDataset(dev_query, dev_lengths, intent_dev, all_out_dev, )
    intent_data_test = MarryUpIntentBatchDataset(test_query, test_lengths, intent_test, all_out_test)

    print('len train dataset {}'.format(len(intent_data_train) if intent_data_train else 0))
    print('len dev dataset {}'.format(len(intent_data_dev) if intent_data_dev else 0))
    print('len test dataset {}'.format(len(intent_data_test)))

    intent_dataloader_train = DataLoader(intent_data_train, batch_size=args.bz) if intent_data_train else None
    intent_dataloader_dev = DataLoader(intent_data_dev, batch_size=args.bz) if intent_data_dev else None
    intent_dataloader_test = DataLoader(intent_data_test, batch_size=args.bz)

    # pretrained_embed = load_glove_embed('../data/{}/'.format(args.dataset), args.embed_dim)
    pretrained_embed = np.random.random((len(t2i) - 1, args.embed_dim))

    # for padding
    pretrain_embed_extend = np.append(pretrained_embed, np.zeros((1, args.embed_dim), dtype=np.float), axis=0)

    model = IntentMarryUp(
        pretrained_embed=pretrain_embed_extend,
        config=args,
        label_size=len(in2i),
    )

    criterion = torch.nn.CrossEntropyLoss()
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0)
    if args.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)

    if torch.cuda.is_available():
        model = model.cuda()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('ALL TRAINABLE PARAMETERS: {}'.format(pytorch_total_params))
    acc_dev_init, avg_loss_dev_init, p, r = val_marry(model, intent_dataloader_dev, 0, 'DEV', args, logger)
    # TEST
    acc_test_init, avg_loss_test_init, p, r = val_marry(model, intent_dataloader_test, 0, 'TEST', args, logger)

    best_dev_acc = acc_dev_init
    counter = 0
    best_epoch = 0
    best_dev_model = deepcopy(model)
    # when no training data, just run a test.
    if not intent_dataloader_train: args.epoch = 0

    for epoch in range(1, args.epoch + 1):
        avg_loss = 0
        acc = 0

        pbar_train = tqdm(intent_dataloader_train)
        pbar_train.set_description("TRAIN EPOCH {}".format(epoch))

        model.train()
        for batch in pbar_train:

            optimizer.zero_grad()

            x = batch['x']
            label = batch['i'].view(-1)
            lengths = batch['l']
            re_tag = batch['re']

            if torch.cuda.is_available():
                x = x.cuda()
                lengths = lengths.cuda()
                label = label.cuda()
                re_tag = re_tag.cuda()

            scores = model(x, lengths, re_tag)

            loss_cross_entropy = criterion(scores, label)

            if args.model_type == 'MarryUp':
                loss = loss_cross_entropy

            elif args.model_type == 'KnowledgeDistill':
                softmax_scores = torch.log_softmax(scores, 1)
                softmax_re_tag_teacher = torch.softmax(re_tag, 1)
                loss_KL = torch.nn.KLDivLoss()(softmax_scores, softmax_re_tag_teacher)
                loss = loss_cross_entropy * args.l1 + loss_KL * (
                        1 - args.l1)  # in KD, l1 stands for the alpha controlling to learn from true / imitate teacher

            elif args.model_type == 'PR':
                log_softmax_scores = torch.log_softmax(scores, 1)
                softmax_scores = torch.softmax(scores, 1)
                product_term = torch.exp(
                    re_tag - 1) * args.l2  # in PR, l2 stands for the regularization term, higher l2, harder rule constraint
                teacher_score = torch.mul(softmax_scores, product_term)
                softmax_teacher = torch.softmax(teacher_score, 1)
                loss_KL = torch.nn.KLDivLoss()(log_softmax_scores, softmax_teacher)
                loss = loss_cross_entropy * args.l1 + loss_KL * (
                        1 - args.l1)  # in PR, l1 stands for the alpha controlling to learn from true / imitate teacher

            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

            acc += (scores.argmax(1) == label).sum().item()

            pbar_train.set_postfix_str("{} - total right: {}, total loss: {}".format('TRAIN', acc, loss))

        acc = acc / len(intent_data_train)
        avg_loss = avg_loss / len(intent_data_train)
        # print("{} Epoch: {} | ACC: {}, LOSS: {}".format('TRAIN', epoch, acc, avg_loss))
        logger.add("{} Epoch: {} | ACC: {}, LOSS: {}".format('TRAIN', epoch, acc, avg_loss))

        # DEV
        acc_dev, avg_loss_dev, p, r = val_marry(model, intent_dataloader_dev, epoch, 'DEV', args, logger)

        counter += 1  # counter for early stopping

        if (acc_dev is None) or (acc_dev > best_dev_acc):
            counter = 0
            best_epoch = epoch
            best_dev_acc = acc_dev
            best_dev_model = deepcopy(model)

        if counter > args.early_stop:
            break

    best_dev_test_acc, avg_loss_test, best_dev_test_p, best_dev_test_r \
        = val_marry(best_dev_model, intent_dataloader_dev, best_epoch, 'TEST', args, logger)

    model_dir = os.path.dirname(__file__) + "/model/{}/".format(args.rnn)
    if not os.path.exists(model_dir):
        mkdir(model_dir)

    model_save_path = os.path.dirname(__file__) + "/model/{}/{}-{}/{}-{}".format(
        args.rnn, args.dataset, args.dataset_spilt, args.dataset, args.dataset_spilt
    )
    model_save_dir = os.path.dirname(__file__) + "/model/{}/{}-{}/".format(args.rnn, args.dataset, args.dataset_spilt)
    if not os.path.exists(model_save_dir):
        mkdir(model_save_dir)
    print("SAVING MODEL {} .....".format(model_save_path))
    torch.save(model.state_dict(), model_save_path + '.model')
    torch.save(model, model_save_path + '.m')

    return acc_dev_init, acc_test_init, best_dev_acc, best_dev_test_acc, best_dev_test_p, best_dev_test_r, logger.record
