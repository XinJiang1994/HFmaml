import json
import numpy as np
import tensorflow as tf
from tqdm import trange

from tensorflow.contrib import rnn

from flearn.utils.model_utils import batch_data
from flearn.utils.language_utils import line_to_indices
from flearn.utils.tf_utils import graph_size, process_grad

# with open('flearn/models/sent140/embs.json', 'r') as inf:
#     embs = json.load(inf)
# id2word = embs['vocab']
# word2id = {v: k for k, v in enumerate(id2word)}
# word_emb = np.array(embs['emba'])
#
#
# def process_x(raw_x_batch, max_words=25):
#     x_batch = [e[4] for e in raw_x_batch]
#     x_batch = [line_to_indices(e, word2id, max_words) for e in x_batch]
#     x_batch = np.array(x_batch)
#     return x_batch
#
#
# def process_y(raw_y_batch):
#     y_batch = [1 if e == '4' else 0 for e in raw_y_batch]
#     y_batch = np.array(y_batch)
#
#     return y_batch


class Model(object):

    def __init__(self, model_params, opt1, seed=1):
        # params
        self.seq_len = model_params[0]
        self.num_classes = model_params[1]
        self.n_hidden = model_params[2]

        self.dim_input = 1
        self.dim_output =1
        self.channels =1

        #self.emb_arr = word_emb
        self.dim_hidden = num_filters # need this value
        self.forward = self.forward_conv
        self.construct_weights = self.construct_conv_weights
        self.optimizer = tf.train.GradientDescentOptimizer(opt1)

        # self.construct_weights = self.construct_fc_weights
        # self.forward = self.forward_fc
        # self.optimizer = tf.train.GradientDescentOptimizer(opt1)

        # create computation graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123 + seed)
            self.features, self.labels, self.train_op, self.grads, self.eval_metric_ops, self.loss = self.create_model(
                self.optimizer)
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)

        # find memory footprint and compute cost of the model
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops

    def create_model(self, optimizer):
        features = tf.placeholder(tf.int32, [None, self.seq_len], name='features')
        labels = tf.placeholder(tf.int64, [None, ], name='labels')

        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
            else:
                # Define the weights
                self.weights = weights = self.construct_weights()

        outputs = self.forward(features, weights, reuse=True)  # only reuse on the first iter
        loss = self.xent(outputs, labels)

        # embs = tf.Variable(self.emb_arr, dtype=tf.float32, trainable=False)
        # x = tf.nn.embedding_lookup(embs, features)
        #
        # stacked_lstm = rnn.MultiRNNCell(
        #     [rnn.BasicLSTMCell(self.n_hidden) for _ in range(2)])
        # outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32)
        fc1 = tf.layers.dense(inputs=outputs[:, -1, :], units=30)
        pred = tf.squeeze(tf.layers.dense(inputs=fc1, units=1))

        #loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=pred)
        # optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        correct_pred = tf.equal(tf.to_int64(tf.greater(pred, 0)), labels)
        eval_metric_ops = tf.count_nonzero(correct_pred)

        return features, labels, train_op, grads, eval_metric_ops, loss

    def set_params(self, model_params=None):
        if model_params is not None:
            with self.graph.as_default():
                all_vars = tf.trainable_variables()
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    # def get_gradients(self, data, model_len):
    #
    #     grads = np.zeros(model_len)
    #     num_samples = len(data['y'])
    #     processed_samples = 0
    #
    #     if num_samples < 50:
    #         input_data = process_x(data['x'])
    #         target_data = process_y(data['y'])
    #         with self.graph.as_default():
    #             model_grads = self.sess.run(self.grads,
    #                                         feed_dict={self.features: input_data, self.labels: target_data})
    #             grads = process_grad(model_grads)
    #         processed_samples = num_samples
    #
    #     else:  # calculate the grads in a batch size of 50
    #         for i in range(min(int(num_samples / 50), 4)):
    #             input_data = process_x(data['x'][50 * i:50 * (i + 1)])
    #             target_data = process_y(data['y'][50 * i:50 * (i + 1)])
    #             with self.graph.as_default():
    #                 model_grads = self.sess.run(self.grads,
    #                                             feed_dict={self.features: input_data, self.labels: target_data})
    #
    #             flat_grad = process_grad(model_grads)
    #             grads = np.add(grads, flat_grad)  # this is the average in this batch
    #
    #         grads = grads * 1.0 / min(int(num_samples / 50), 4)
    #         processed_samples = min(int(num_samples / 50), 4) * 50
    #
    #     return processed_samples, grads

    def get_gradients(self, data, model_len):

        grads = np.zeros(model_len)
        num_samples = len(data['y'])

        with self.graph.as_default():
            model_grads = self.sess.run(self.grads,
                feed_dict={self.features: data['x'], self.labels: data['y']})
            grads = process_grad(model_grads)

        return num_samples, grads

    # def solve_inner(self, data, num_epochs=1, batch_size=32):
    #     '''
    #     Args:
    #         data: dict of the form {'x': [list], 'y': [list]}
    #     Return:
    #         comp: number of FLOPs computed while training given data
    #         update: list of np.ndarray weights, with each weight array
    #     corresponding to a variable in the resulting graph
    #     '''
    #
    #     for _ in trange(num_epochs, desc='Epoch: ', leave=False):
    #         for X, y in batch_data(data, batch_size):
    #             input_data = process_x(X, self.seq_len)
    #             target_data = process_y(y)
    #             with self.graph.as_default():
    #                 self.sess.run(self.train_op,
    #                               feed_dict={self.features: input_data, self.labels: target_data})
    #     soln = self.get_params()
    #     comp = num_epochs * (len(data['y']) // batch_size) * batch_size * self.flops
    #     return soln, comp
    def solve_inner(self, data, num_epochs):
        '''Solves local optimization problem'''
        batch_size = len(data['y'])
        for _ in trange(num_epochs, desc='Epoch: ', leave=False, ncols=120):
            for X, y in batch_data(data, batch_size):
                with self.graph.as_default():
                    self.sess.run(self.train_op,
                        feed_dict={self.features: X, self.labels: y})
        soln = self.get_params()
        comp = num_epochs * (len(data['y'])//batch_size) * batch_size * self.flops
        return soln, comp

    def test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        x_vecs = process_x(data['x'], self.seq_len)
        labels = process_y(data['y'])
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
                                              feed_dict={self.features: x_vecs, self.labels: labels})
        return tot_correct, loss

    def close(self):
        self.sess.close()


    def construct_conv_weights(self):
        weights = {}

        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
        k = 3

        weights['conv1'] = tf.get_variable('conv1', [k, k, self.channels, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv2'] = tf.get_variable('conv2', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv3'] = tf.get_variable('conv3', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv4'] = tf.get_variable('conv4', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]))
        # if FLAGS.datasource == 'miniimagenet':
        #     # assumes max pooling
        #     weights['w5'] = tf.get_variable('w5', [self.dim_hidden*5*5, self.dim_output], initializer=fc_initializer)
        #     weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        # else:
        weights['w5'] = tf.Variable(tf.random_normal([self.dim_hidden, self.dim_output]), name='w5')
        weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        return weights

    def xent(self, pred, label):
        # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
        return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label)


    def forward_conv(self, inp, weights, reuse=False):
        # reuse is for the normalization parameters.
        # channels = self.channels
        # inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])

        #with tf.variable_scope("cnn") as scope:
        def weight_variable(shape, name):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name=name)

        def bias_variable(shape, name):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=name)

        def conv2d(x, W, stride, padding='SAME'):
            return tf.nn.conv2d(x, W, strides=stride, padding=padding)

        W_conv1 = weight_variable([3, 3, 3, 32], name='W_conv1')
        b_conv1 = bias_variable([32], name='b_conv1')
        h_conv1 = conv2d(inp, W_conv1, stride=[1, 1, 1, 1, 1], padding='VALID')
        h_conv1 = tf.nn.bias_add(h_conv1, b_conv1)
        h_conv1 = lrelu(h_conv1)
        h_pool1 = tf.nn.max_pool3d(h_conv1, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='VALID',
                                   name='Max_pool1')

        W_conv2 = weight_variable([3, 3, 32, 64], name='W_conv2')
        b_conv2 = bias_variable([64], name='b_conv2')
        h_conv2 = conv2d(h_pool1, W_conv2, stride=self.stride)
        h_conv2 = tf.nn.bias_add(h_conv2, b_conv2)

        h_pool2 = tf.nn.max_pool3d(h_conv2, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='VALID',
                                   name='Max_pool2')

        W_conv3 = weight_variable([3, 3, 64, 128], name='W_conv3')
        b_conv3 = bias_variable([64], name='b_conv3')
        h_conv3 = conv2d(h_pool2, W_conv3, stride=self.stride)
        h_conv3 = tf.nn.bias_add(h_conv3, b_conv3)
        h_pool3 = tf.nn.max_pool3d(h_conv3, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='VALID',
                                   name='Max_pool3')

        W_conv4 = weight_variable([3, 3, 64, 128], name='W_conv4')
        b_conv4 = bias_variable([64], name='b_conv4')
        h_conv4 = conv2d(h_pool3, W_conv4, stride=self.stride)
        h_conv4 = tf.nn.bias_add(h_conv4, b_conv4)
        pred = tf.nn.avg_pool(h_conv4, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='VALID',
                                   name='avg_pool1')
        return pred


def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)