import cPickle as pickle
import re
import numpy as np
from collections import Counter

import tensorflow as tf

def get_vocabulary(data_path):
    vocab_count = pickle.load(
        open(data_path + '/word_dict_count_nosp.pkl', mode='rb'))
    vocabulary = Counter(vocab_count)
    vocab_pairs = sorted(vocabulary.items())
    alphabet, _ = list(zip(*vocab_pairs))
    return dict(zip(alphabet, xrange(1, len(alphabet) + 1)))

def char_to_id(vocabulary, char):
    return vocabulary.get(char, 0)

# loads complete dataset
def get_raw_data(data_path, vocabulary, model_type):
    with open(data_path, mode='r') as f:
        train_data, valid_data, test_data = __get_data(f,
                                                       vocabulary,
                                                       model_type)

    print("data closed")
    return train_data, valid_data, test_data

def __get_data(f, vocabulary, model_type):
    data_size = 0
    for _ in f:
        data_size += 1
    f.seek(0)  # get back to the start

    train_data_size = (int)(0.7 * data_size)
    valid_data_size = (int)(0.15 * data_size)

    print(train_data_size)
    print(valid_data_size)
    print('data to vectors')
    train_data = []
    valid_data = []
    test_data = []
    if model_type == 'word':
        for i, line in enumerate(f):
            ids = [char_to_id(vocabulary, char.lower())
                   for char in line.strip().split()]
            if i < train_data_size:
                train_data.extend(ids)
            elif i >= train_data_size and i < train_data_size + valid_data_size:
                valid_data.extend(ids)
            elif i >= train_data_size + valid_data_size and i < train_data_size + 2 * valid_data_size:
                test_data.extend(ids)
    elif model_type == 'char':
        for i, line in enumerate(f):
            ids = [char_to_id(vocabulary, char.lower())
                   for char in list(unicode(line, 'utf-8').strip())]
            if i < train_data_size:
                train_data.extend(ids)
            elif i >= train_data_size and i < train_data_size + valid_data_size:
                valid_data.extend(ids)
            elif i >= train_data_size + valid_data_size and i < train_data_size + 2 * valid_data_size:
                test_data.extend(ids)
    return train_data, valid_data, test_data

# generates data for network (train, valid or test)
def data_generator(data, batch_size, unroll_size):
    data = np.array(data, dtype=np.int32)
    data_size = len(data)
    # num of units (letters, words, ...) in a batch
    num_of_batches = (int)(data_size / batch_size)
    batched_data = np.zeros([batch_size, num_of_batches], dtype=np.int32)
    print(batch_size, unroll_size, num_of_batches)
    for i in xrange(batch_size):
        batched_data[i] = data[i * num_of_batches:(i + 1) * num_of_batches]

    epoch_size = (int)((num_of_batches - 1) / unroll_size)

    for i in xrange(epoch_size):
        x = batched_data[:, i * unroll_size:(i + 1) * unroll_size]
        y = batched_data[:, i * unroll_size + 1:(i + 1) * unroll_size + 1]
        yield x, y
