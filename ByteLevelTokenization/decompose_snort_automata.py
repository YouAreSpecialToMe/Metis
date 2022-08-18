import os
import pickle
from fsa_to_tensor import dfa_to_tensor
import argparse
from load_dataset import load_classification_dataset
from Metis.utils.data import decompose_tensor_split


def decompose_automata(args):
    merged_automata = pickle.load(
        open(os.path.dirname(__file__) + '/data/snort/{}/automata/{}.pkl'.format(args.dataset,
                                                                                 args.automata_name), 'rb'))

    print('AUTOMATA TO TENSOR')
    print('Total States: {}'.format(len(merged_automata['states'])))
    # first load vocabs
    dataset = load_classification_dataset(args)
    # print(dataset['data'])
    word2idx = dataset['t2i']
    print(word2idx)
    language_tensor, state2idx, wildcard_mat, language = dfa_to_tensor(merged_automata, word2idx)
    complete_tensor = language_tensor + wildcard_mat

    print('DECOMPOSE SPLIT AUTOMATA')

    for random_state in range(1):
        print('DECOMPOSING RANK: {}, TENSOR SIZE: {}'.format(args.rank, language_tensor.shape))
        V_embed_split, D1_split, D2_split, rec_error = \
            decompose_tensor_split(language_tensor, language, word2idx, args.rank,
                                   random_state=random_state, n_iter_max=30, init=args.init)

        save_dict = {
            'automata': merged_automata,
            'V': V_embed_split,
            'D1': D1_split,
            'D2': D2_split,
            'language': language,
            'wildcard_mat': wildcard_mat,
        }
        pickle.dump(save_dict, open(
            os.path.dirname(__file__) + '/data/snort/{}/automata/automata.{}.{}.pkl'.format(args.dataset,
                                                                                            args.dataset,
                                                                                            args.rank), 'wb'))

    print('FINISHED')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='chat', help="dataset name")
    parser.add_argument('--automata_name', type=str, default='all',
                        help="automata name prefix")
    parser.add_argument('--rank', type=int, default=200, help="rank")
    parser.add_argument('--init', type=str, default='svd', help="initialization")
    parser.add_argument('--dataset_spilt', type=float, default=1, help="rate of using labeled data")

    args = parser.parse_args()
    assert args.init in ['svd', 'random']

    decompose_automata(args)
