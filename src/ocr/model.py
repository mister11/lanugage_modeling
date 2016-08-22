import tensorflow as tf
from tensorflow.models.rnn import rnn
import numpy as np

class Model:
    def __init__(self, is_train, is_test, vocab_size, config, model_type):
        self.batch_size = config.batch_size
        self.unroll_size = config.unroll_size
        self.hidden_size = config.hidden_size
        self.num_of_layers = config.num_of_layers
        self.grad_clip = config.grad_clip
        self.learning_rate = config.learning_rate
        self.model_type = model_type
        self.vocab_size = vocab_size
        if is_test:
            self.batch_size = 1
            self.unroll_size = 1

        self.cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] *
                                                self.num_of_layers)
        self.init_state = self.cell.zero_state(self.batch_size, tf.float32)

        self.input_data = tf.placeholder(tf.int32, [self.batch_size,
                                                    self.unroll_size])
        self.labels = tf.placeholder(tf.int32, [self.batch_size,
                                                self.unroll_size])

        embedding = tf.get_variable('embedding', [self.vocab_size,
                                                  self.hidden_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        inputs = tf.split(1, self.unroll_size, inputs)
        inputs = [tf.squeeze(inp, [1]) for inp in inputs]
        outputs, state = rnn.rnn(self.cell, inputs,
                                 initial_state=self.init_state)

        output = tf.reshape(tf.concat(1, outputs), [-1, self.hidden_size])
        self.softmax_w = tf.get_variable('softmax_w', [self.hidden_size,
                                                       self.vocab_size])
        self.softmax_b = tf.get_variable('softmax_b', [self.vocab_size])

        logits = tf.matmul(output, self.softmax_w) + self.softmax_b
        self.probabs = tf.nn.softmax(logits)

        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self.labels, [-1])],
            [tf.ones([self.batch_size * self.unroll_size])],
            self.vocab_size
        )
        self.cost = tf.reduce_sum(loss) / self.batch_size
        self.final_state = state

        if is_train:
            lr = tf.Variable(self.learning_rate, trainable=False)
            trainable_vars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost,
                                                           trainable_vars),
                                              self.grad_clip)
            optimizer = tf.train.AdamOptimizer(lr)
            self.train_op = optimizer.apply_gradients(zip(grads,
                                                          trainable_vars))

    def sample(self, sess, vocabulary, seed, sample_size=100):
        chars = sorted(vocabulary, key=lambda w: vocabulary[w])
        state = self.cell.zero_state(1, tf.float32).eval()
        for word in seed:
            x = np.zeros((1, 1))
            x[0, 0] = vocabulary.get(word, 0)
            [state] = sess.run([self.final_state],
                               {self.input_data: x, self.init_state: state})

        sample_text = seed
        curr_word = seed
        for _ in range(sample_size):
            x = np.zeros((1, 1))
            x[0, 0] = vocabulary.get(curr_word, 0)
            [state, probabs] = sess.run([self.final_state, self.probabs],
                                        {self.input_data: x,
                                        self.init_state: state})
            word = np.random.choice(chars, p=probabs.flatten())
            curr_word = word
            sample_text += word
        return sample_text

    def eval_candidates(self, sess, all_candidates, vocabulary, out):
        init_state = self.cell.zero_state(1, tf.float32).eval()
        states = [(init_state, None, 0, '')]
        #print('total candidates: ' + str(len(all_candidates)))
        for i, word_candidates in enumerate(all_candidates):
            new_states = []
            if i % (len(all_candidates) * 0.1) == 0:
                out.write("progress: " + str(i) + "\n")
            #out.write(str(word_candidates))
            maxx = max([state[2] for state in states])
            if len(states) > 1000:
                states = list(filter(lambda x: x[2] > 0.99 * maxx, states))
            out.write("num of states: " + str(len(states)) + "\n")
            out.flush()
            for j, state in enumerate(states):
                if j % 200 == 0:
                    out.write("processing state: " + str(j) + "\n")
                    out.flush()
                if type(word_candidates) == list:
                    for splits in word_candidates:
                        #print("SPLITS: ", splits)
                        curr_state = state
                        p = 0
                        for word in splits:
                            #print('WORD: ', word)
                            if curr_state[1] is not None and word in vocabulary:
                                p += curr_state[1][vocabulary.get(word)]
                            x = np.zeros((1, 1))
                            x[0, 0] = vocabulary.get(word, 0)
                            [new_state, probabs] = sess.run([self.final_state,
                                                             self.probabs],
                                                            {self.input_data: x,
                                                             self.init_state:
                                                             state[0]})
                            probabs = probabs.flatten()
                            curr_state = (new_state, probabs, curr_state[2] + p,
                                          curr_state[3] + '|' + word)
                        new_states.append(curr_state)
                else:
                    word = word_candidates
                    p = 0
                    # if there is old probabs distribution, find probab for
                    # current word, if word not in vocab, probab is 0
                    if state[1] is not None and word in vocabulary:
                        p = state[1][vocabulary.get(word)]
                    x = np.zeros((1, 1))
                    x[0, 0] = vocabulary.get(word, 0)
                    [new_state, probabs] = sess.run([self.final_state,
                                                     self.probabs],
                                                    {self.input_data: x,
                                                     self.init_state: state[0]})
                    probabs = probabs.flatten()
                    new_states.append((new_state, probabs, state[2] + p,
                                       state[3] + '|' + word))
            states = list(new_states)
        ret_val = [(c, d) for _, _, c, d in states]
        del states
        del new_states
        return ret_val
