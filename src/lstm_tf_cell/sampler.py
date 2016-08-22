# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import argparse
import imp
from collections import Counter
import random

import dataset
from model import Model

out = open('sample_h.txt', mode='a')

def main(args):
    config = imp.load_source('config', args.config)
    vocabulary = dataset.get_vocabulary(args.data_dir)
    vocabulary = Counter(vocabulary)
    vocab_size = len(vocabulary)
    with tf.Graph().as_default(), tf.Session() as sess:
        var_initializer = tf.random_uniform_initializer(config.init_min,
                                                        config.init_max)

        with tf.variable_scope('model', reuse=False,
                               initializer=var_initializer):
            model = Model(False, True, vocab_size, config, 'test')

        tf.initialize_all_variables().run()
        saver = tf.train.Saver()

        ckpt_dir = './ls_h_w_512_20/'
        latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
        if latest_ckpt:
            saver.restore(sess, latest_ckpt)
        cands = ['kako', 'kamo', 'zašto', 'gdje', 'zar']
        seeds = ['when', 'where', 'what', 'how', 'great', 'less']
        for i in range(20):
            out.write("SAMPLE: \n")
            out.write(model.sample(sess, vocabulary,
                                        np.random.choice(cands)))
            out.write("\n\n")
            out.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir',
                        help="data directory",
                        required=True,
                        default="./data/")
    parser.add_argument('-m', '--model',
                        help="model type",
                        choices=["char", "word"],
                        required=True,
                        default="word")
    parser.add_argument('-c', '--config',
                        help="configuration file",
                        required=True,
                        default="./config.py")
    main(parser.parse_args())
