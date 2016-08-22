# -*- coding: utf-8 -*-
from os import listdir
import tensorflow as tf
import dataset
import argparse
import imp
from collections import Counter
from itertools import combinations

from model import Model

out = open('./ocr_out.txt', mode='a')

def possible_splits(word, max_splits=2):
    pick_from = range(1, len(word))
    to_pick = range(1, max_splits + 1)
    for pick in to_pick:
        for combs in combinations(pick_from, pick):
            yield [word[i:j] for i, j in zip((0,) + combs, combs + (None,))]

def get_candidates(line, vocabulary):
    line = line.strip().lower()
    if len(line) == 0:
        return []
    tokens = line.split()
    all_candidates = []
    for word in tokens:
        candidates = []
        if word not in vocabulary:
            poss_splits = possible_splits(word)
            candidates.append([word])
            for w in poss_splits:
                candidates.append(w)
        else:
            candidates = word
        all_candidates.append(candidates)
    return all_candidates

def eval_candidates(model, candidates, vocabulary, out):
    return model.eval_candidates(candidates, vocabulary, out)

def run_corrector(sess, model, files, vocabulary):
    k = open('./sentences.txt', mode='a')
    for file in files:
        out.write(file + "\n")
        out.flush()
        with open(file, mode='r') as f:
            for line in f:
                out.write("LINE: " + line + "\n")
                out.flush()
                # list of possible sentences
                out.write('getting candidates' + "\n")
                candidates = get_candidates(line, vocabulary)
                if len(candidates) == 0:
                    continue
                out.write('evaluating candidates' + "\n")
                out.flush()
                best_sentences = model.eval_candidates(sess, candidates,
                                                       vocabulary, out)
                out.write('eval done' + "\n")
                best_sentences = sorted(best_sentences, key=lambda x: -x[0])
                k.write("BESTs: " + str(best_sentences[0]) + "\n\n")
                k.flush()

def main(args):
    config = imp.load_source('config', args.config)
    vocabulary = dataset.get_vocabulary(args.data_dir)
    vocabulary = Counter(vocabulary)
    # print(test(['izlazio', [['jeiz'], ['j', 'eiz'], ['je', 'iz'],
    #                         ['jei', 'z'], ['j', 'e', 'iz'],
    #                         ['j', 'ei', 'z'], ['je', 'i', 'z'],
    #                         ['j', 'e', 'i', 'z']], 'nije'], vocabulary))
    # return
    vocab_size = len(vocabulary)
    datasets_root_path = args.data_dir + '/error/'
    #file_list = listdir(datasets_root_path)
    #file_list = [datasets_root_path + name for name in file_list]
    file_list = ['./test_file_ocr.txt']
    with tf.Graph().as_default(), tf.Session() as sess:
        var_initializer = tf.random_uniform_initializer(config.init_min,
                                                        config.init_max)

        with tf.variable_scope('model', reuse=False,
                               initializer=var_initializer):
            train_model = Model(True, False, vocab_size, config, 'train')
        #
        # with tf.variable_scope('model', reuse=True,
        #                        initializer=var_initializer):
        #     valid_model = Model(False, False, vocab_size, config, 'valid')
        #
        tf.initialize_all_variables().run()
        saver = tf.train.Saver()
        print("init done")
        ckpt_dir = args.data_dir + "/saved_model/"
        latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
        if latest_ckpt:
            saver.restore(sess, latest_ckpt)
        print('running corrector')
        run_corrector(sess, train_model, file_list,
                      vocabulary)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir',
                        help="data directory",
                        required=True,
                        default="./data/")
    parser.add_argument('-c', '--config',
                        help="configuration file",
                        required=True,
                        default="./config.py")
    main(parser.parse_args())
