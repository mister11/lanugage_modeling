# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import argparse
import imp
from os import listdir
import random
from collections import Counter

import dataset
from model import Model

out = open('test_hrwac_chunks_chars.txt', mode='a')

def run_epoch(sess, model, data_filenames_list, vocabulary,
              model_type, eval_op):
    total_cost = 0.0
    iters = 0
    state = model.init_state.eval()
    for i, filename in enumerate(data_filenames_list):
        data1, data2, data3 = dataset.get_raw_data(
            filename,
            vocabulary,
            model_type)
        data = data1 + data2 + data3
        epoch_size = ((len(data) // model.batch_size) - 1) // model.unroll_size
        for step, (x, y) in enumerate(
            dataset.data_generator(data,
                                   model.batch_size,
                                   model.unroll_size)):
            l, probs, cost, state, _ = sess.run([model.l, model.probabs,
                                                 model.cost, model.final_state,
                                                 eval_op],
                                                {model.input_data: x,
                                                 model.labels: y,
                                                 model.init_state: state})
            total_cost += cost
            iters += model.unroll_size
            if (step + 1) % 2000 == 0:
                out.write(str(y) + "\n")
                out.write("--> Intermediate perplexity: " +
                          str((np.exp(total_cost / iters))))
                out.write("  (" + str((1.0 * step / epoch_size)) + ")")
                out.write("\n")
                out.flush()

    return np.exp(total_cost / iters)


def main(args):
    config = imp.load_source('config', args.config)
    vocabulary = dataset.get_vocabulary(args.data_dir)
    vocabulary = Counter(vocabulary)
    vocab_size = len(vocabulary)
    datasets_root_path = args.data_dir + '/text/chunks/'
    test_list = listdir(datasets_root_path + 'test/')
    test_list = [datasets_root_path + 'test/' + name for name in test_list]
    with tf.Graph().as_default(), tf.Session() as sess:
        var_initializer = tf.random_uniform_initializer(config.init_min,
                                                        config.init_max)

        with tf.variable_scope('model', reuse=False,
                               initializer=var_initializer):
            model = Model(False, True, vocab_size, config, 'test')


        tf.initialize_all_variables().run()
        saver = tf.train.Saver()

        ckpt_dir = './local_saver/'
        latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
        if latest_ckpt:
            out.write("Restored model: " + str(latest_ckpt) + "\n")
            saver.restore(sess, latest_ckpt)

        out.write('Calculating perplexity...' + "\n")

        perplexity = run_epoch(sess, model, test_list,
                               vocabulary, args.model,
                               tf.no_op())
        out.write("Final perplexity: " + str(train_perplexity) + "\n\n")

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
