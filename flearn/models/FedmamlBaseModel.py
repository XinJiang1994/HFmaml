import tensorflow as tf
import numpy as np
from flearn.utils.model_utils import batch_data
from tensorflow.python import debug as tf_debug
from tqdm import trange


### This is an implenmentation of Hessian Free maml meta learning algirithm propoesed by Sheng Yue####
### THis is Base model of the algorithm which implenments the optimization part####
###

class BaseModel(object):
    def __init__(self, params):
        self.k = 0
        self.alpha = params['alpha']
        self.beta = params['beta']
        # self.num_local_updates = params['num_local_updates']
        self.seed = params['seed']
        self.graph = tf.Graph()
        self.theta_kp1 = None
        self.optimizer2 = tf.train.GradientDescentOptimizer(self.beta)

        with self.graph.as_default():
            self.sess = tf.Session(graph=self.graph)
            # self.sess = tf_debug.TensorBoardDebugWrapperSession(self.sess, "TC174611125:32005")
            self.weights = self.construct_weights()  # weights is a list
            tf.set_random_seed(123 + self.seed)
            self.features_train, self.labels_train, self.features_test, self.labels_test = self.get_input()
            self.eval_metric_ops, self.predictions_test, self.loss, self.train_loss, self.test_op, self.fast_vars = self.optimize()
            self.saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())

    def optimize(self):
        with self.graph.as_default():
            w_names = [x.name.split(':', 1)[0] for x in self.weights]
            logits_train = self.forward_func(self.features_train, self.weights, w_names, reuse=True)
            self.logits_train=logits_train
            loss_train = self.loss_func(logits_train, self.labels_train)
            grad_w = tf.gradients(loss_train, self.weights)
            phy = [val - self.alpha * grad for grad, val in zip(grad_w, self.weights)]

            logits_test = self.forward_func(inp=self.features_test, weights=phy, w_names=w_names, reuse=True)
            loss_test = self.loss_func(logits_test, self.labels_test)
            # grad_Ltest2phy=tf.gradients(loss_test,phy)


            logits_final=self.forward_func(inp=self.features_test,weights=self.weights,w_names=w_names)
            predictions_test = {
                "classes": tf.argmax(input=tf.nn.softmax(logits_final, name="softmax_tensor"), axis=1),
                "probabilities": tf.nn.softmax(logits_final, name="softmax_tensor")
            }
            pred_test = tf.argmax(input=logits_test, axis=1)
            self.test_acc = tf.reduce_mean(
                tf.cast(tf.equal(tf.cast(tf.argmax(input=self.labels_test, axis=1), dtype=tf.float32),
                                 tf.cast(pred_test, dtype=tf.float32)), dtype=tf.float32))

            g_and_v = self.optimizer2.compute_gradients(loss_test)
            grads_test, _ = zip(*g_and_v)
            test_op = self.optimizer2.apply_gradients(g_and_v, global_step=tf.train.get_global_step())

            eval_metric_ops = tf.reduce_mean(
                tf.cast(tf.equal(tf.cast(tf.argmax(input=self.labels_test, axis=1), dtype=tf.float32),
                                 tf.cast(predictions_test["classes"], dtype=tf.float32)), dtype=tf.float32))
        return eval_metric_ops, predictions_test, loss_test, loss_train, test_op, phy

    def get_input(self):
        '''
        :return:the placeholders of input: features_train,labels_train,features_test,labels_test
        '''
        pass

    def forward_func(self, inp, weights, w_names, reuse=False):

        '''
        :param inp: input
        :param weights: theta
        :param reuse:
        :return: model y
         when overload this function you should make w=dict(zip(w_names,weights))
        '''
        pass

    def construct_weights(self):
        '''
        :return:weights
        '''
        pass

    def loss_func(self, logits, label):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label)
        return tf.reduce_mean(losses)

    def solve_inner(self, train_data, test_data, num_epochs):
        for _ in range(num_epochs):
            X_train = train_data['x']
            y_train = train_data['y']
            X_test = test_data['x']
            y_test = test_data['y']
            with self.graph.as_default():
                self.sess.run(self.test_op,
                              feed_dict={self.features_train: X_train, self.labels_train: y_train,
                                         self.features_test: X_test, self.labels_test: y_test})
        soln = self.get_params()
        return soln

    def fast_adapt(self, train_data, num_epochs):
        batch_size_train = len(train_data['y'])
        # batch_size_test = len(test_data['y'])
        for _ in trange(num_epochs, desc='Epoch: ', leave=False, ncols=120):
            for X_train, y_train in batch_data(train_data, batch_size_train):
                # for X_test, y_test in batch_data(test_data, batch_size_test):
                with self.graph.as_default():
                    soln = self.sess.run(self.fast_vars,
                                         feed_dict={self.features_train: X_train, self.labels_train: y_train})
                    self.set_params(soln)
        return soln

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

    def get_phy(self,train_data):
        with self.graph.as_default():
            phy_val = self.sess.run(self.fast_vars,feed_dict={self.features_train: train_data['x'], self.labels_train: train_data['y']})
        return phy_val

    def get_logits_train(self,train_data):
        with self.graph.as_default():
            logits = self.sess.run(self.logits_train,feed_dict={self.features_train: train_data['x'], self.labels_train: train_data['y']})
        return logits

    def get_features_train(self,train_data):
        with self.graph.as_default():
            features_train = self.sess.run(self.features_train,feed_dict={self.features_train: train_data['x'], self.labels_train: train_data['y']})
        return features_train


    def test(self, train_data, test_data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''

        with self.graph.as_default():
            acc, loss, train_loss = self.sess.run([self.test_acc, self.loss, self.train_loss],
                                                  feed_dict={self.features_train: train_data['x'],
                                                             self.labels_train: train_data['y'],
                                                             self.features_test: test_data['x'],
                                                             self.labels_test: test_data['y']})
        return acc, loss

    def test_test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        with self.graph.as_default():
            acc, loss_train, preds = self.sess.run(
                [self.eval_metric_ops, self.train_loss, self.predictions_test["classes"]],
                feed_dict={self.features_train: data['x'], self.labels_train: data['y'],
                           self.features_test: data['x'], self.labels_test: data['y']})
        return acc, loss_train, preds


    def target_acc_while_train(self, train_data, test_data):
        target_test_acc = self.sess.run(self.test_acc,
                            feed_dict={self.features_train: train_data['x'], self.labels_train: train_data['y'],
                                       self.features_test: test_data['x'], self.labels_test: test_data['y']})
        return target_test_acc
