import numpy as np
import tensorflow as tf
from tqdm import trange
import operator

from tensorflow.contrib.layers.python import layers as tf_layers
from flearn.utils.model_utils import batch_data
from flearn.utils.tf_utils import graph_size
from flearn.utils.tf_utils import process_grad



class Model(object):
    """assumes that images are 28px by 28px"""

    def __init__(self, num_classes, opt1, opt2, train_data, test_data, seed=1, num_local_updates=1):
        # we can define two optimizer here for two learning rate
        # optimizer is defined in fedavg,
        # learner is used in fedbase, learner is attribute of model
        # if we define model with two optimizer, learner has two
        # params, input learning rate, not optimizer
        #tf.global_variables_initializer().run()
        self.num_classes = num_classes
        self.construct_weights = self.construct_fc_weights
        self.optimizer1 = tf.train.GradientDescentOptimizer(opt1)
        self.optimizer2 = tf.train.GradientDescentOptimizer(opt2)
        self.forward = self.forward_fc
        #init_op = tf.initialize_all_variables()
        # creat computation graph
        self.graph = tf.Graph()


        #self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123+seed)
            self.sess = tf.Session(graph=self.graph)
            self.sess.run(tf.global_variables_initializer())
            self.features_train, self.labels_train, self.features_test, self.labels_test, self.test_op,\
            self.grads, self.eval_metric_ops, self.loss, self.train_loss, self.fast_vars = self.creat_model(opt1, num_local_updates)
            self.saver = tf.train.Saver()
            #self.sess.run(init_op)
            #self.sess = tf.Session(graph=self.graph)

        # find memory footprint and compute cost of the model
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops


    def creat_model(self, opt1, num_local_updates=1):

        features_train = tf.placeholder(tf.float32, shape=[None, 784], name='features_train')
        labels_train = tf.placeholder(tf.float32, shape=[None, 10], name='labels_train')
        features_test = tf.placeholder(tf.float32, shape=[None, 784], name='features_test')
        labels_test = tf.placeholder(tf.float32, shape=[None, 10], name='labels_test')


        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
            else:
                # Define the weights
                self.weights = weights = self.construct_weights()


        logits_train = self.forward(features_train, weights, reuse = True)
        # logits_train = tf.layers.dense(inputs=features_train, units=self.num_classes, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
        # logits_train shape:  (?, 10)
        predictions_train = {
                "classes": tf.argmax(input=logits_train, axis=1),
                "probabilities": tf.nn.softmax(logits_train, name="softmax_tensor")
        }
        loss_train = self.xent(logits_train,labels_train)
        loss_train = tf.reduce_mean(loss_train)
        grads = tf.gradients(loss_train, list(weights.values()))
        gradients = dict(zip(weights.keys(), grads))
        fast_vars = dict(zip(weights.keys(), [weights[key] - opt1*gradients[key] for key in weights.keys() ]))

        # grads_and_vars = self.optimizer1.compute_gradients(loss_train)
        # grads,vars = zip(*grads_and_vars)
        # fast_vars = tuple(map(operator.add, vars, tuple((-opt1)*x for x in grads)))
        # # fast_vars = tuple(map(operator.add, vars, tuple((-opt1)*x for x in grads)))
        # print('fast_vars: ',fast_vars)
        # print('*********************************************************')
        # fast_vars:  (<tf.Tensor 'add:0' shape=(784, 10) dtype=float32>, <tf.Tensor 'add_1:0' shape=(10,) dtype=float32>)

        for j in range(num_local_updates-1):
            print('current j: {}  and num_local_updates: {}'.format(j, num_local_updates))
            logits_train = self.forward(features_train, fast_vars, reuse=True)
            loss_train = self.xent(logits_train, labels_train)
            loss_train = tf.reduce_mean(loss_train)
            grads = tf.gradients(loss_train, list(fast_vars.values()))
            gradients = dict(zip(fast_vars.keys(), grads))
            fast_vars = dict(zip(fast_vars.keys(), [fast_vars[key] - opt1 * gradients[key] for key in fast_vars.keys()]))

        grads_and_vars = self.optimizer1.compute_gradients(loss_train)
        # grads, _ = zip(*grads_and_vars)

        # train_op = self.optimizer1.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        logits_test = self.forward(inp=features_test,weights=fast_vars, reuse=True)


        predictions_test = {
            "classes": tf.argmax(input=logits_test, axis=1),
            "probabilities": tf.nn.softmax(logits_test, name="softmax_tensor")
        }
        loss_test = self.xent(logits_test,labels_test)
        loss_test = tf.reduce_mean(loss_test)
        # print('loss_test shape: ',loss_test.shape)


        g_and_v = self.optimizer2.compute_gradients(loss_test)
        grads_test,_ = zip(*g_and_v)
        test_op = self.optimizer2.apply_gradients(g_and_v, global_step=tf.train.get_global_step())
        eval_metric_ops = tf.count_nonzero(tf.equal(tf.cast(tf.argmax(input=labels_test, axis=1),dtype=tf.float32), tf.cast(predictions_test["classes"],dtype=tf.float32)))
        return features_train, labels_train, features_test, labels_test, test_op, grads_test, eval_metric_ops, loss_test, loss_train, fast_vars



    def solve_inner(self, train_data, test_data, num_epochs):
        batch_size_train = len(train_data['y'])
        batch_size_test = len(test_data['y'])
        for _ in trange(num_epochs, desc='Epoch: ', leave=False, ncols=120):
            for X_train, y_train in batch_data(train_data, batch_size_train):
                for X_test, y_test in batch_data(test_data, batch_size_test):
                    with self.graph.as_default():
                        self.sess.run(self.test_op,
                                  feed_dict={self.features_train: X_train, self.labels_train: y_train,
                                             self.features_test: X_test, self.labels_test: y_test})
        soln = self.get_params()
        comp = num_epochs * (len(test_data['y']) // batch_size_test) * batch_size_test * self.flops
        return soln, comp

    def fast_adapt(self, train_data, num_epochs):
        batch_size_train = len(train_data['y'])
       #batch_size_test = len(test_data['y'])
        for _ in trange(num_epochs, desc='Epoch: ', leave=False, ncols=120):
            for X_train, y_train in batch_data(train_data, batch_size_train):
                #for X_test, y_test in batch_data(test_data, batch_size_test):
                    with self.graph.as_default():
                        soln=self.sess.run(self.fast_vars,
                                  feed_dict={self.features_train: X_train, self.labels_train: y_train})
                                            # self.features_test: X_test, self.labels_test: y_test})
        #soln = self.get_params()
        comp = num_epochs * (len(train_data['y']) // batch_size_train) * batch_size_train * self.flops
        return soln, comp


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


    def get_gradients(self, train_data, test_data, model_len):

        #grads_t = np.zeros(model_len)
        num_samples = len(test_data['y'])

        batch_size_train = len(train_data['y'])
        batch_size_test = len(test_data['y'])
        #for _ in trange(num_epochs=1, desc='Epoch: ', leave=False, ncols=120):
        for X_train, y_train in batch_data(train_data, batch_size_train):
                for X_test, y_test in batch_data(test_data, batch_size_test):
                    with self.graph.as_default():
                        model_grads = self.sess.run(self.grads,
                                      feed_dict={self.features_train: X_train, self.labels_train: y_train,
                                                 self.features_test: X_test, self.labels_test: y_test})

        #with self.graph.as_default():
        #    model_grads = self.sess.run(self.grads,
        #                                feed_dict={self.features_test: data['x'], self.labels_test: data['y']})
                        grads_t = process_grad(model_grads)

        return num_samples, grads_t


    def construct_fc_weights(self):
        weights = {}
        weights['w1'] = tf.Variable(tf.truncated_normal([784, self.num_classes], stddev=0.01))
        #weights['b1'] = tf.Variable(tf.zeros([self.num_classes]))
        weights['b1'] = tf.Variable(tf.zeros([self.num_classes]))
        return weights


    def test(self, train_data, test_data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        batch_size_train =  len(train_data['y'])
        batch_size_test = len(test_data['y'])
        #tot_correct = []
        tot_correct = np.zeros(np.size(test_data['y']))
        loss = np.zeros(np.size(test_data['y']))
        # for _ in trange(num_epochs=1, desc='Epoch: ', leave=False, ncols=120):
        for X_train, y_train in batch_data(train_data, batch_size_train):
            for X_test, y_test in batch_data(test_data, batch_size_test):
                with self.graph.as_default():
                    tot_correct, loss, train_loss = self.sess.run([self.eval_metric_ops, self.loss, self.train_loss],
                                              feed_dict={self.features_train: X_train, self.labels_train: y_train,
                                                         self.features_test: X_test, self.labels_test: y_test})
        return tot_correct, loss

    def test_test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        with self.graph.as_default():
            tot_correct, loss_train = self.sess.run([self.eval_metric_ops, self.train_loss],
                                                    feed_dict={self.features_train: data['x'], self.labels_train: data['y'],
                                                               self.features_test: data['x'], self.labels_test: data['y']})
        return tot_correct, loss_train


    def close(self):
        self.sess

    # def construct_weight(self):
    #     weights={}
    #     weights['w1'] = tf.Variable(tf.truncated_normal([784, self.num_classes], stddev=0.01))
    #     weights['b1'] = tf.Variable(tf.zeros([self.num_classes]))
    #     return  weights

    def forward_fc(self,inp, weights, reuse = False):
        hidden = tf.matmul(inp, weights['w1']) + weights['b1']
        return hidden

    def normalize(self,inp, activation, reuse, scope):
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)

    def mse(self, pred, label):
        pred = tf.reshape(pred, [-1])
        label = tf.reshape(label, [-1])
        return tf.reduce_mean(tf.square(pred - label))

    def xent(self, pred, label):
        # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
        return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label)

    def zeroth_loss(self, test_data):
        batch_size_test = len(test_data['y'])
        for X_test, y_test in batch_data(test_data, batch_size_test):
            with self.graph.as_default():
               zero_loss =  self.sess.run(self.train_loss,
                              feed_dict={self.features_train: X_test, self.labels_train: y_test})

        return zero_loss


    def test_train(self, train_data):
        batch_size_train = len(train_data['y'])
        for X_train, y_train in batch_data(train_data, batch_size_train):
            with self.graph.as_default():
                            self.sess.run(self.fast_vars,
                              feed_dict={self.features_train: X_train, self.labels_train: y_train})
        fast_vars=self.get_params()

        return fast_vars

