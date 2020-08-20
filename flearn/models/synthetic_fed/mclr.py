import numpy as np
import tensorflow as tf
from tqdm import trange


from flearn.utils.model_utils import batch_data
from flearn.utils.tf_utils import graph_size
from flearn.utils.tf_utils import process_grad


class Model(object):
    '''
    Assumes that images are 28px by 28px
    '''

    def __init__(self, num_classes, opt1, seed=1, num_local_updates=1):

        # params
        self.num_classes = num_classes
        self.construct_weights = self.construct_fc_weights
        self.forward = self.forward_fc
        self.optimizer = tf.train.GradientDescentOptimizer(opt1)
        # create computation graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123 + seed)
            self.features, self.labels, self.train_op, \
            self.grads, self.eval_metric_ops, self.loss, self.loss_train, self.fast_vars = self.create_model(opt1,
                                                                                                             num_local_updates)
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)

        # find memory footprint and compute cost of the model
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops

    def create_model(self, opt1, num_local_updates=1):
        """Model function for Logistic Regression."""
        features = tf.placeholder(tf.float32, shape=[None, 60], name='features')
        labels = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
        Reduction = tf.losses.Reduction


        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
            else:
                # Define the weights
                self.weights = weights = self.construct_weights()

        # logits = tf.layers.dense(inputs=features, units=self.num_classes, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
        logits = self.forward(features, weights, reuse=True)
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits, reduction=Reduction.SUM_OVER_BATCH_SIZE)
        #loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        #print("loss_a", loss_a)
        #loss = tf.reduce_mean(loss)
        #loss= tf.reduce_sum(loss) / tf.to_float(len(labels))
        #loss = tf.reduce_sum(loss) / 16
        loss_train = loss

        grads = tf.gradients(loss, list(weights.values()))
        gradients = dict(zip(weights.keys(), grads))
        fast_vars = dict(zip(weights.keys(), [weights[key] - opt1 * gradients[key] for key in weights.keys()]))

        # grads_and_vars = self.optimizer.compute_gradients(loss)
        # grads, _ = zip(*grads_and_vars)

        for j in range(num_local_updates - 1):
            print('current j: {}  and num_local_updates: {}'.format(j, num_local_updates))
            logits = self.forward(features, fast_vars, reuse=True)
            predictions = {
                "classes": tf.argmax(input=logits, axis=1),
                "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }
            loss = self.xent(logits, labels)
            #loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            loss = tf.reduce_mean(loss)
            grads = tf.gradients(loss, list(fast_vars.values()))
            gradients = dict(zip(fast_vars.keys(), grads))
            fast_vars = dict(
                zip(fast_vars.keys(), [fast_vars[key] - opt1 * gradients[key] for key in fast_vars.keys()]))

        grads_and_vars = self.optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)

        train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
        # train_op = fast_vars
        # eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        eval_metric_ops = tf.count_nonzero(tf.equal(tf.cast(tf.argmax(input=labels, axis=1), dtype=tf.float32),
                                                    tf.cast(predictions["classes"], dtype=tf.float32)))
        return features, labels, train_op, grads, eval_metric_ops, loss, loss_train, fast_vars

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

    def get_gradients(self, data, model_len):

        grads = np.zeros(model_len)
        num_samples = len(data['y'])

        with self.graph.as_default():
            model_grads = self.sess.run(self.grads,
                                        feed_dict={self.features: data['x'], self.labels: data['y']})
            grads = process_grad(model_grads)

        return num_samples, grads

    def solve_inner(self, data, num_epochs):
        '''Solves local optimization problem'''
        batch_size = len(data['y'])
        for _ in trange(num_epochs, desc='Epoch: ', leave=False, ncols=120):
            for X, y in batch_data(data, batch_size):
                with self.graph.as_default():
                    self.sess.run(self.train_op,
                                  feed_dict={self.features: X, self.labels: y})
        soln = self.get_params()
        comp = num_epochs * (len(data['y']) // batch_size) * batch_size * self.flops
        return soln, comp

    def fast_adapt(self, data, num_epochs):
        '''Solves local optimization problem'''
        batch_size = len(data['y'])
        for _ in trange(num_epochs, desc='Epoch: ', leave=False, ncols=120):
            for X, y in batch_data(data, batch_size):
                with self.graph.as_default():
                    soln = self.sess.run(self.fast_vars,
                                         feed_dict={self.features: X, self.labels: y})
        # soln = self.get_params()
        # print("shape: ", soln.shape)
        comp = num_epochs * (len(data['y']) // batch_size) * batch_size * self.flops
        return soln, comp

    def test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
                                              feed_dict={self.features: data['x'], self.labels: data['y']})
        print('loss: ', loss)
        return tot_correct, loss

    def test_test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        with self.graph.as_default():
            tot, loss_train = self.sess.run([self.eval_metric_ops, self.loss_train],
                                            feed_dict={self.features: data['x'], self.labels: data['y']})
        return tot, loss_train

    def final_test(self, train_data, test_data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
                                              feed_dict={self.features: train_data['x'], self.labels: train_data['y']})

        with self.graph.as_default():
            test_loss = self.sess.run(self.loss_train,
                                      feed_dict={self.features: test_data['x'], self.labels: test_data['y']})

        return tot_correct, loss, test_loss

    def close(self):
        self.sess.close()

    def xent(self, pred, label):
        # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
        return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label)

    def construct_fc_weights(self):
        weights = {}
        weights['w1'] = tf.Variable(tf.truncated_normal([60, self.num_classes], stddev=0.01))
        # weights['b1'] = tf.Variable(tf.zeros([self.num_classes]))
        weights['b1'] = tf.Variable(tf.zeros([self.num_classes]))
        return weights

    def forward_fc(self, inp, weights, reuse=False):
        hidden = tf.matmul(inp, weights['w1']) + weights['b1']
        return hidden

    def zeroth_loss(self, test_data):
        batch_size_test = len(test_data['y'])
        for X_test, y_test in batch_data(test_data, batch_size_test):
            with self.graph.as_default():
                zero_loss = self.sess.run(self.loss_train,
                                          feed_dict={self.features: X_test, self.labels: y_test})

        return zero_loss

    def zeroth_loss_w(self, test_data):
        weight = self.get_params()
        logits = self.forward(test_data['x'], weight, reuse=True)
        zero_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=test_data['y'])
        zero_loss = tf.reduce_mean(zero_loss)

        # batch_size_test = len(test_data['y'])
        # for X_test, y_test in batch_data(test_data, batch_size_test):
        #     with self.graph.as_default():
        #         zero_loss = self.sess.run(self.loss_train,
        #                                   feed_dict={self.features: X_test, self.labels: y_test})

        return zero_loss
    # def __init__(self, num_classes, optimizer, seed=1):
    #
    #     # params
    #     self.num_classes = num_classes
    #
    #     # create computation graph
    #     self.graph = tf.Graph()
    #     with self.graph.as_default():
    #         tf.set_random_seed(123+seed)
    #         self.features, self.labels, self.train_op, self.grads, self.eval_metric_ops, self.loss, self.pred = self.create_model(optimizer)
    #         self.saver = tf.train.Saver()
    #     self.sess = tf.Session(graph=self.graph)
    #
    #     # find memory footprint and compute cost of the model
    #     self.size = graph_size(self.graph)
    #     with self.graph.as_default():
    #         self.sess.run(tf.global_variables_initializer())
    #         metadata = tf.RunMetadata()
    #         opts = tf.profiler.ProfileOptionBuilder.float_operation()
    #         self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops
    #
    # def create_model(self, optimizer):
    #     """Model function for Logistic Regression."""
    #     features = tf.placeholder(tf.float32, shape=[None, 60], name='features')
    #     labels = tf.placeholder(tf.int64, shape=[None,], name='labels')
    #     logits = tf.layers.dense(inputs=features, units=self.num_classes, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
    #     predictions = {
    #         "classes": tf.argmax(input=logits, axis=1),
    #             "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    #         }
    #     loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    #
    #     grads_and_vars = optimizer.compute_gradients(loss)
    #     grads, _ = zip(*grads_and_vars)
    #     train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
    #     eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
    #     return features, labels, train_op, grads, eval_metric_ops, loss, predictions["classes"]
    #
    # def set_params(self, model_params=None):
    #     if model_params is not None:
    #         with self.graph.as_default():
    #             all_vars = tf.trainable_variables()
    #             for variable, value in zip(all_vars, model_params):
    #                 variable.load(value, self.sess)
    #
    # def get_params(self):
    #     with self.graph.as_default():
    #         model_params = self.sess.run(tf.trainable_variables())
    #     return model_params
    #
    # def get_gradients(self, data, model_len):
    #
    #     grads = np.zeros(model_len)
    #     num_samples = len(data['y'])
    #
    #     with self.graph.as_default():
    #         model_grads = self.sess.run(self.grads,
    #             feed_dict={self.features: data['x'], self.labels: data['y']})
    #         grads = process_grad(model_grads)
    #
    #     return num_samples, grads
    #
    # def solve_inner(self, data, num_epochs=1, batch_size=32):
    #     '''Solves local optimization problem'''
    #     for _ in trange(num_epochs, desc='Epoch: ', leave=False, ncols=120):
    #         for X, y in batch_data(data, batch_size):
    #             with self.graph.as_default():
    #                 _, pred = self.sess.run([self.train_op, self.pred],
    #                     feed_dict={self.features: X, self.labels: y})
    #     soln = self.get_params()
    #     comp = num_epochs * (len(data['y'])//batch_size) * batch_size * self.flops
    #     return soln, comp
    #
    # def test(self, data):
    #     '''
    #     Args:
    #         data: dict of the form {'x': [list], 'y': [list]}
    #     '''
    #     with self.graph.as_default():
    #         tot_correct, loss, pred = self.sess.run([self.eval_metric_ops, self.loss, self.pred],
    #             feed_dict={self.features: data['x'], self.labels: data['y']})
    #         #print("predictions on test data: {}\n".format(pred))
    #     return tot_correct, loss
    #
    # def close(self):
    #     self.sess.close()
