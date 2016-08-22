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

out = open('output_ocr.txt', mode='a')

def run_epoch(sess, model, data_filenames_list, vocabulary,
              model_type, eval_op, epoch_saver):
    total_cost = 0.0
    iters = 0
    state = model.init_state.eval()
    random.shuffle(data_filenames_list)
    for i, filename in enumerate(data_filenames_list):
        out.write(filename + "\n")
	out.flush()
        data1, data2, data3 = dataset.get_raw_data(
            filename,
            vocabulary,
            model_type)
	data = data1 + data2 + data3
        epoch_size = ((len(data) // model.batch_size) - 1) // model.unroll_size
        for step, (x, y) in enumerate(dataset.data_generator(data,
                                                             model.batch_size,
                                                             model.unroll_size)
                                      ):
	    cost, state, _ = sess.run([model.cost, model.final_state, eval_op],
                                      {model.input_data: x,
                                       model.labels: y,
                                       model.init_state: state})
            total_cost += cost
            iters += model.unroll_size

           
            out.write("--> Perplexity: " +
                          str((np.exp(total_cost / iters))))
            out.write("  (" + str((1.0 * step / epoch_size)) + ")")
            out.write("\n")
            out.flush()
        epoch_saver.save(sess, './ocr_saver/model-' + model.model_type +
                         '.ckpt', global_step=i)

    return np.exp(total_cost / iters)


def main(args):
    config = imp.load_source('config', args.config)
    vocabulary = dataset.get_vocabulary(args.data_dir)
    vocabulary = Counter(vocabulary)
    vocab_size = len(vocabulary)
    datasets_root_path = args.data_dir + '/correct/'
    train_list = listdir(datasets_root_path + 'train/')
    train_list = [datasets_root_path + 'train/' + name for name in train_list]
    valid_list = listdir(datasets_root_path + 'valid/')
    valid_list = [datasets_root_path + 'valid/' + name for name in valid_list]
    with tf.Graph().as_default(), tf.Session() as sess:
        var_initializer = tf.random_uniform_initializer(config.init_min,
                                                        config.init_max)

        with tf.variable_scope('model', reuse=False,
                               initializer=var_initializer):
            train_model = Model(True, False, vocab_size, config, 'train')

        with tf.variable_scope('model', reuse=True,
                               initializer=var_initializer):
            valid_model = Model(False, False, vocab_size, config, 'valid')

        tf.initialize_all_variables().run()
        saver = tf.train.Saver()
        epoch_saver = tf.train.Saver(max_to_keep=50)

        ckpt_dir = args.data_dir + "/saved_model/"
        latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
        if latest_ckpt:
            saver.restore(sess, latest_ckpt)

        old_valid_perplexity = None
        for i in xrange(config.num_of_epochs):
            out.write("Epoch: " + str(i + 1))
            out.write("\n")
            out.write("-" * 20)
            out.write("\n")
            out.flush()

            train_perplexity = run_epoch(sess, train_model, train_list,
                                         vocabulary, args.model,
                                         train_model.train_op,
                                         epoch_saver)
            out.write("Train perplexity: " + str(train_perplexity))
            out.write("\n")
            out.write("Epoch: " + str(i + 1))
            out.write("\n")
            out.write("-" * 20)
            out.write("\n")
            out.flush()
            valid_perplexity = run_epoch(sess, valid_model, valid_list,
                                         vocabulary, args.model,
                                         tf.no_op(), epoch_saver)
            #if (old_valid_perplexity is None or
            #        valid_perplexity < old_valid_perplexity):
            #    old_valid_perplexity = valid_perplexity
            #    saver.save(sess, args.data_dir + '/saved_model/' +
            #               model.ckpt, global_step=i)

            out.write("Valid perplexity: " + str(valid_perplexity))
            out.write("\n")
            out.flush()
        out.write("END")

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
