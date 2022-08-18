import warnings
import argparse
from SoftTree import *
import time
import random
import math
import pickle
import os

random.seed(2020)
warnings.filterwarnings('ignore')

def ProduceSoftLabel(h_l, sco, k=2):
    temp = 1
    hard_label = np.zeros([h_l.shape[0], len(np.unique(h_l[:, -1]))])
    for i in range(np.shape(h_l)[0]):
        hard_label[i][int(h_l[i, -1])] = 1
    for i in range(sco.shape[0]):
        sco[i][0], sco[i][1] = max(-10, sco[i][0]), max(-10, sco[i][1])
        sco[i][0], sco[i][1] = min(10, sco[i][0]), min(10, sco[i][1])
        sco[i][0], sco[i][1] = math.exp(sco[i][0]/temp)/(math.exp(sco[i][0]/temp) + math.exp(sco[i][1]/temp)), math.exp(sco[i][1]/temp)/(math.exp(sco[i][0]/temp) + math.exp(sco[i][1]/temp))

    soft_label = (sco + hard_label*k) / (k+1)
    return soft_label


def ProduceBinaryInput(data_vec):
    BV = []
    for v in data_vec:
        if v >= 2**7:
            BV.append(1)
            v -= 2**7
        else:
            BV.append(0)
        if v >= 2**6:
            BV.append(1)
            v -= 2**6
        else:
            BV.append(0)
        if v >= 2**5:
            BV.append(1)
            v -= 2**5
        else:
            BV.append(0)
        if v >= 2**4:
            BV.append(1)
            v -= 2**4
        else:
            BV.append(0)
        if v >= 2**3:
            BV.append(1)
            v -= 2**3
        else:
            BV.append(0)
        if v >= 2**2:
            BV.append(1)
            v -= 2**2
        else:
            BV.append(0)
        if v >= 2**1:
            BV.append(1)
            v -= 2**1
        else:
            BV.append(0)
        if v >= 2**0:
            BV.append(1)
            v -= 2**0
        else:
            BV.append(0)
    return BV


def ConstructTree(data_train, data_label, feature_attr, snort_name, tree_name, r, file_list='tree',
                  min_sample_leaf=15, feature_num='sqrt', train_sample=15000):
    clf = SoftTreeClassifier(n_features=feature_num, min_sample_leaf=min_sample_leaf)

    clf.fit(data_train, data_label, feature_attr)
    if os.path.exists(os.path.join(file_list, snort_name, str(r))) == False:
        os.makedirs(os.path.join(file_list, snort_name, str(r)))
    with open(os.path.join(file_list, snort_name, str(r), tree_name), 'wb') as f:  # open file with write-mode
        pickle.dump(clf, f)


def TestForest(snort_name, data_test, data_label, r):
    file_name = 'tree'
    dir_ = os.listdir(os.path.join(file_name, snort_name, str(r)))
    clf_list = []
    for file in dir_[0:15]:
        if os.path.isdir(os.path.join(file_name, snort_name, str(r), file)):
            continue
        with open(os.path.join(file_name, snort_name, str(r), file), 'rb') as f:
            clf_list.append(pickle.load(f))  # read file and build object
    result = []
    start_time = time.time()
    for clf_ in clf_list:
        if result == []:
            result = clf_.predict(data_test)
        else:
            result += clf_.predict(data_test)
    end_time = time.time()
    print(start_time-end_time)
    final_res = []
    for r_ in result:
        if r_ > max(len(clf_list)//2-2, 0):
            final_res.append(1)
        else:
            final_res.append(0)
    acc = accuracy(np.array(final_res), data_label)
    print("  soft decision tree:", acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for data label')
    parser.add_argument('--mode', type=str, help='train or test')
    parser.add_argument('--snort_name', type=str, help='snort name')
    parser.add_argument('--label_r', type=float, help='labeled ratio')
    args = parser.parse_args()
    mode = args.mode
    labeled_ratio = args.label_r

    snort_name = args.snort_name

    data_train = np.load(file='softLabel/'+snort_name+'-'+str(labeled_ratio)+'/train_x.npy')
    data_train = np.array([v[1:] for v in data_train])
    # data_train = np.array([ProduceBinaryInput(v) for v in data_train])
    hard_label = np.load(file='softLabel/'+snort_name+'-'+str(labeled_ratio)+'/train_label.npy').reshape(-1,1)

    scores = np.load(file='softLabel/'+snort_name+'-'+str(labeled_ratio)+'/train_score.npy')
    correct = 0
    wrong = 0
    for num in range(len(scores)):
        if scores[num][0] >= scores[num][1] and hard_label[num][0] == 0:
            correct += 1
        elif scores[num][0] < scores[num][1] and hard_label[num][0] == 1:
            correct += 1
        else:
            wrong += 1
    print(correct/len(scores))

    data_test = np.load(file='softLabel/'+snort_name+'-'+str(labeled_ratio)+'/test_x.npy')
    data_test = np.array([v[1:] for v in data_test])
    # data_test = np.array([ProduceBinaryInput(v) for v in data_test])
    hard_label_test = np.load(file='softLabel/'+snort_name+'-'+str(labeled_ratio)+'/test_label.npy').reshape(-1,1)
    scores_test = np.load(file='softLabel/'+snort_name+'-'+str(labeled_ratio)+'/test_score.npy')

    feature_attr = ['c'] * data_train.shape[1]
    print(data_train.shape)
    print(data_test.shape)

    if mode == 'train':
        from multiprocessing import Process, Pool
        from random import sample
        snort_list = [
            'chat',
            'ftp',
            'games',
            'malware',
            'misc',
            'netbios',
            'p2p',
            'policy',
            'telnet',
            'tftp',
            'web_client',
        ]
        for snort_name in snort_list:
            print(snort_name)

            data_train = np.load(file='softLabel/' + snort_name + '-' + str(labeled_ratio) + '/train_x.npy')
            data_train = np.array([v[1:] for v in data_train])
            # data_train = np.array([ProduceBinaryInput(v) for v in data_train])
            hard_label = np.load(file='softLabel/' + snort_name + '-' + str(labeled_ratio) + '/train_label.npy').reshape(-1,
                                                                                                                         1)

            scores = np.load(file='softLabel/' + snort_name + '-' + str(labeled_ratio) + '/train_score.npy')
            correct = 0
            wrong = 0
            for num in range(len(scores)):
                if scores[num][0] >= scores[num][1] and hard_label[num][0] == 0:
                    correct += 1
                elif scores[num][0] < scores[num][1] and hard_label[num][0] == 1:
                    correct += 1
                else:
                    wrong += 1
            print(correct / len(scores))

            data_test = np.load(file='softLabel/' + snort_name + '-' + str(labeled_ratio) + '/test_x.npy')
            data_test = np.array([v[1:] for v in data_test])
            # data_test = np.array([ProduceBinaryInput(v) for v in data_test])
            hard_label_test = np.load(
                file='softLabel/' + snort_name + '-' + str(labeled_ratio) + '/test_label.npy').reshape(-1, 1)
            scores_test = np.load(file='softLabel/' + snort_name + '-' + str(labeled_ratio) + '/test_score.npy')

            feature_attr = ['d'] * data_train.shape[1]
            print(data_train.shape)
            print(data_test.shape)

            tree_num = 15
            t_num = 20000
            p = Pool(tree_num)
            start_time = time.time()
            all_num = data_train.shape[0]
            hard_label_input = []
            for _ in range(all_num):
                if _ <= all_num * labeled_ratio:
                    hard_label_input.append(hard_label[_])
                else:
                    if scores[_][0] >= scores[_][1]:
                        hard_label_input.append([0])
                    else:
                        hard_label_input.append([1])
            soft_label = ProduceSoftLabel(np.array(hard_label_input), scores)
            for i in range(tree_num):
                # print(i)
                random.seed()
                ind_ = sample(range(data_train.shape[0]), t_num)
                data_train_ = np.array([data_train[_] for _ in ind_])
                soft_label_ = np.array([soft_label[_] for _ in ind_])
                p.apply_async(ConstructTree, args=(data_train_, soft_label_, feature_attr, snort_name, str(i), labeled_ratio))
            p.close()
            p.join()
            end_time = time.time()
            print(end_time-start_time)

    if mode == 'test':
        TestForest(snort_name, data_test, hard_label_test, labeled_ratio)

    if mode == 'dt':
        all_num = data_train.shape[0]
        hard_label_input = []
        for _ in range(all_num):
            if _ <= all_num * labeled_ratio:
                hard_label_input.append(hard_label[_])
            else:
                if scores[_][0] > scores[_][1]:
                    hard_label_input.append([0])
                else:
                    hard_label_input.append([1])

        from sklearn.tree import DecisionTreeClassifier
        all_num = data_train.shape[0]
        t_num = int(all_num * labeled_ratio)
        clf = DecisionTreeClassifier(max_features="sqrt", min_samples_leaf=15)
        clf.fit(data_train, hard_label_input)
        clf.fit(data_train[0:t_num], hard_label_input[0:t_num])
        res = clf.predict(data_test)
        acc = accuracy(np.array(res), hard_label_test)
        print("  decision tree:", acc)

    if mode == 'dtdistill':
        all_num = data_train.shape[0]
        hard_label_input = []
        for _ in range(all_num):
            if _ <= all_num * labeled_ratio:
                hard_label_input.append(hard_label[_])
            else:
                if scores[_][0] >= scores[_][1]:
                    hard_label_input.append([0])
                else:
                    hard_label_input.append([1])

        from sklearn.tree import DecisionTreeClassifier
        all_num = data_train.shape[0]
        t_num = int(all_num * labeled_ratio)
        clf = DecisionTreeClassifier(max_features="sqrt", min_samples_leaf=15)
        clf.fit(data_train, hard_label_input)
        clf.fit(data_train, hard_label_input)
        res = clf.predict(data_test)
        acc = accuracy(np.array(res), hard_label_test)
        print("  decision tree:", acc)

    if mode == 'rf':
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        all_num = data_train.shape[0]
        t_num = int(all_num * labeled_ratio)
        rfc = RandomForestClassifier(random_state=0, n_estimators=7, min_samples_leaf=15, max_features='sqrt',
                                     max_samples=min(t_num, 20000))
        rfc = rfc.fit(data_train[0:t_num], hard_label[0:t_num])
        score_r = rfc.predict(data_test)
        acc = accuracy(score_r, hard_label_test)
        print("  random forest:", acc)

    if mode == 'rfdistill':
        all_num = data_train.shape[0]
        hard_label_input = []
        for _ in range(all_num):
            if _ <= all_num * labeled_ratio:
                hard_label_input.append(hard_label[_])
            else:
                if scores[_][0] >= scores[_][1]:
                    hard_label_input.append([0])
                else:
                    hard_label_input.append([1])
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        all_num = data_train.shape[0]
        t_num = int(all_num * labeled_ratio)
        rfc = RandomForestClassifier(random_state=0, n_estimators=15, min_samples_leaf=15, max_features='sqrt',
                                     max_samples=20000)
        rfc = rfc.fit(data_train, hard_label_input)
        score_r = rfc.predict(data_test)
        acc = accuracy(score_r, hard_label_test)
        print("  random forest:", acc)
