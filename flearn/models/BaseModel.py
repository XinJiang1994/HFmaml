import tensorflow as tf
# tf.set_random_seed(123)
### This is an implenmentation of Hessian Free maml meta learning algirithm propoesed by Sheng Yue####
### THis is Base model of the algorithm which implenments the optimization part####
###
import numpy as np

class BaseModel(object):
    def __init__(self, params):
        # print('@BaseModel line 17 test init')
        self.k = 0
        self.alpha = params['alpha']
        self.rho = params['rho']
        self.w_i = params['w_i']
        self.mu_i = params['mu_i']
        self.seed = params['seed']
        self.graph = tf.Graph()
        self.theta_kp1 = None
        self.optimizer1 = tf.train.GradientDescentOptimizer(self.alpha)

        with self.graph.as_default():
            self.sess = tf.Session(graph=self.graph)
            # self.sess = tf_debug.TensorBoardDebugWrapperSession(self.sess, "TC174611125:32005")
            self.weights = self.construct_weights()  # weights is a list
            self.transfer_weights,self.meta_weights,self.w_names,self.meta_w_names=self.get_meta_weights()
            self.yy_k = self.construct_yy_k()
            # tf.set_random_seed(123 + self.seed)
            self.delta = tf.Variable(1000.0, dtype=tf.float32, trainable=False)
            self.features_train, self.labels_train, self.features_test, self.labels_test = self.get_input()
            self.optimize()
            self.saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())

    def get_meta_weights(self):
        w_names = [x.name.split(':', 1)[0] for x in self.weights]
        transfer_weight_name = ['W_conv1', 'b_conv1','W_conv2', 'b_conv2','W_conv3', 'b_conv3']
        meta_weight_name = [name for name in w_names if name not in transfer_weight_name]
        weights = dict(zip(w_names, self.weights))
        transfer_weights = [weights[name] for name in transfer_weight_name]
        meta_weights = [weights[name] for name in meta_weight_name]
        return transfer_weights, meta_weights,w_names,meta_weight_name

    def optimize(self):
        with self.graph.as_default():
            w_names = [x.name.split(':', 1)[0] for x in self.weights]
            # transfer_weight_name = ['W_conv1', 'b_conv1', 'W_conv2', 'b_conv2']
            meta_weight_name = self.meta_w_names
            weights=dict(zip(w_names, self.weights))
            transfer_weights=self.transfer_weights
            meta_weights = [weights[name] for name in meta_weight_name]

            logits_train = self.forward_func(self.features_train, self.weights, w_names, reuse=True)
            self.logits_train=logits_train
            # print('baseModel line 44 logits_train.shape', logits_train.shape)
            self.train_loss = self.loss_func(logits_train, self.labels_train)

            # print('@BaseModel line 48',self.weights)

            grad_w = tf.gradients(self.train_loss, meta_weights)
            new_meta_weights=[w-self.alpha * g for g,w in zip(grad_w,meta_weights)]
            self.fast_vars = transfer_weights+new_meta_weights

            # g_v = self.optimizer1.compute_gradients(self.train_loss)
            # self.adapt_op = self.optimizer1.apply_gradients(g_v, global_step=tf.train.get_global_step())

            logits_test = self.forward_func(inp=self.features_test, weights=self.fast_vars, w_names=w_names, reuse=True)

            # print("logits:",logits_test)

            self.loss = self.loss_func(logits_test, self.labels_test)

            grad_Ltest2phy = tf.gradients(self.loss,meta_weights)
            # grad_Ltest2phy=[tf.zeros_like(x) for x in self.weights]

            ######### self.grad_Ltest2weight = tf.gradients(self.loss, meta_weights) ************////

            theta_kp1 = meta_weights
            # theta_kp1=[tf.zeros_like(x) for x in self.weights]

            inner_g1 = [th_kp1 + self.delta * gradphy for th_kp1, gradphy in zip(theta_kp1, grad_Ltest2phy)]
            logits_train_p = self.forward_func(self.features_train, transfer_weights+inner_g1, w_names=w_names, reuse=True)
            loss_train_p = self.loss_func(logits_train_p, self.labels_train)
            grad_1 = tf.gradients(loss_train_p, inner_g1)

            inner_g2 = [th_kp1 - self.delta * gradphy for th_kp1, gradphy in zip(theta_kp1, grad_Ltest2phy)]
            logits_train_m = self.forward_func(self.features_train, transfer_weights+inner_g2, w_names=w_names, reuse=True)
            loss_train_m = self.loss_func(logits_train_m, self.labels_train)
            grad_2 = tf.gradients(loss_train_m, inner_g2)
            grad_1 = list(grad_1)
            grad_2 = list(grad_2)

            g_kp1 = [(g1 - g2) / (2 * self.delta) for g1, g2 in zip(grad_1, grad_2)]
            # g_kp1=[tf.zeros_like(x) for x in self.weights]

            self.meta_theta_i_kp1 = [tpkp1 - (yy + self.w_i * (g_phy - self.alpha * gg)) / self.rho for
                            tpkp1, yy, g_phy, gg in zip(theta_kp1, self.yy_k, grad_Ltest2phy, g_kp1)]
            self.theta_i_kp1=transfer_weights + self.meta_theta_i_kp1

            logits_test_final = self.forward_func(inp=self.features_test, weights=self.weights, w_names=w_names,
                                                  reuse=True)

            self.predictions_test = {
                "classes": tf.argmax(input=logits_test_final, axis=1),
                "probabilities": tf.nn.softmax(logits_test_final, name="softmax_tensor")
            }
            pred_test = tf.argmax(input=logits_test, axis=1)
            self.test_acc = tf.reduce_mean(
                tf.cast(tf.equal(tf.cast(tf.argmax(input=self.labels_test, axis=1), dtype=tf.float32),
                                 tf.cast(pred_test, dtype=tf.float32)), dtype=tf.float32))
            self.train_acc = tf.reduce_mean(
                tf.cast(tf.equal(tf.cast(tf.argmax(input=self.labels_train, axis=1), dtype=tf.float32),
                                 tf.cast(tf.argmax(input=logits_train, axis=1), dtype=tf.float32)), dtype=tf.float32))
            self.eval_metric_ops = tf.reduce_mean(
                tf.cast(tf.equal(tf.cast(tf.argmax(input=self.labels_test, axis=1), dtype=tf.float32),
                                 tf.cast(self.predictions_test["classes"], dtype=tf.float32)), dtype=tf.float32))

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

    def construct_yy_k(self):
        with self.graph.as_default():
            tv = self.meta_weights
            # tf.set_random_seed(123)
            yyk = [tf.Variable(tf.truncated_normal(x.shape, stddev=0.01), name='yyk_' + x.name.split(':', 1)[0],
                               dtype=tf.float32, trainable=False) for x in tv]
            # yyk = [tf.Variable(tf.zeros(x.shape), name='yyk_' + x.name.split(':', 1)[0],
            #                    dtype=tf.float32, trainable=False) for x in tv]
        return yyk

    def loss_func(self, logits, label):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label)
        return tf.reduce_mean(losses)


    def solve_inner(self, train_data, test_data, num_epochs):
        self.k += 1
        # for cifar10 delta=1/(k+100)
        #for Fmnist delta=1/(k*10)s
        self.delta.load(1.0 / (self.k+100), self.sess)
        # self.delta.load(1.0 / (self.k*10+100), self.sess)

        X_train = train_data['x']
        y_train = train_data['y']
        X_test = test_data['x']
        y_test = test_data['y']

        with self.graph.as_default():
            # print('@mclr lin 153: theta_kp1 before run', self.theta_kp1)
            meta_thikp1,thikp1 = self.sess.run([self.meta_theta_i_kp1,self.theta_i_kp1],
                                   feed_dict={self.features_train: X_train, self.labels_train: y_train,
                                              self.features_test: X_test, self.labels_test: y_test})
        self.update_yy_k(meta_thikp1)
        # soln = thikp1
        soln = meta_thikp1
        yyk = self.get_yyk()
        # yyk=[np.zeros_like(x) for x in soln]
        return soln, yyk

    def fast_adapt(self, train_data, num_epochs):
        for i in range(num_epochs):
            with self.graph.as_default():
                soln=self.sess.run(self.fast_vars,
                              feed_dict={self.features_train: train_data['x'], self.labels_train: train_data['y']})
                # self.set_params(soln)
        return soln

    def receive_global_theta(self, model_params=None):
        self.theta_kp1 = model_params
        if model_params is not None:
            with self.graph.as_default():
                all_vars = tf.trainable_variables()
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)

    def set_params(self, model_params=None):
        if model_params is not None:
            with self.graph.as_default():
                all_vars = self.weights
                assert len(all_vars) == len(model_params), "set params error len(all_vars)!=len(model_params)"
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    def get_logits_train(self,train_data):
        with self.graph.as_default():
            logits = self.sess.run(self.logits_train,feed_dict={self.features_train: train_data['x'], self.labels_train: train_data['y']})
        return logits

    def get_features_train(self,train_data):
        with self.graph.as_default():
            features_train = self.sess.run(self.features_train,feed_dict={self.features_train: train_data['x'], self.labels_train: train_data['y']})
        return features_train

    def get_phy(self,train_data):
        with self.graph.as_default():
            phy_val = self.sess.run(self.fast_vars,feed_dict={self.features_train: train_data['x'], self.labels_train: train_data['y']})
        return phy_val

    def get_yyk(self):
        with self.graph.as_default():
            yy = self.sess.run(self.yy_k)
        return yy

    def set_yyk(self, vals):
        with self.graph.as_default():
            for y, v in zip(self.yy_k, vals):
                y.load(v, self.sess)

    def update_yy_k(self, meta_thikp1):
        yy_kp1s = []
        yy_ks = self.get_yyk()
        theta_kp1_all = dict(zip(self.w_names, self.theta_kp1))
        theta_kp1_meta=[theta_kp1_all[name] for name in self.meta_w_names]

        for yy_k, theta_kp1_i, theta_kp1 in zip(yy_ks, meta_thikp1, theta_kp1_meta):
            # print(yy_k.shape,theta_kp1_i.shape,theta_kp1.shape)
            # exit(0)
            yy_kp1s.append(yy_k + self.rho * (theta_kp1_i - theta_kp1))
        self.set_yyk(yy_kp1s)

    def test(self, train_data, test_data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''

        # print(train_data)

        with self.graph.as_default():
            acc_train, acc_test, loss, train_loss = self.sess.run(
                [self.train_acc, self.test_acc, self.loss, self.train_loss],
                feed_dict={self.features_train: train_data['x'],
                           self.labels_train: train_data['y'],
                           self.features_test: test_data['x'],
                           self.labels_test: test_data['y']})
        return acc_train, acc_test, loss

    def test_test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        with self.graph.as_default():
            acc, loss_train, preds = self.sess.run(
                [self.eval_metric_ops, self.train_loss, self.predictions_test['probabilities']],
                feed_dict={self.features_train: data['x'], self.labels_train: data['y'],
                           self.features_test: data['x'], self.labels_test: data['y']})
        return acc, loss_train, preds

    def get_gradient_phy_w(self, train_data, test_data):
        X_train = train_data['x']
        y_train = train_data['y']
        X_test = test_data['x']
        y_test = test_data['y']

        with self.graph.as_default():
            # print('@mclr lin 153: theta_kp1 before run', self.theta_kp1)
            grads = self.sess.run(self.grad_Ltest2weight,
                                  feed_dict={self.features_train: X_train, self.labels_train: y_train,
                                             self.features_test: X_test, self.labels_test: y_test})
        return grads

    def get_w_names(self):
        return self.w_names,self.meta_w_names

    def get_param_names(self):
        w_names = [x.name.split(':', 1)[0] for x in self.weights]
        return w_names

    def target_acc_while_train(self, train_data, test_data):
        target_test_acc = self.sess.run(self.test_acc,
                            feed_dict={self.features_train: train_data['x'], self.labels_train: train_data['y'],
                                       self.features_test: test_data['x'], self.labels_test: test_data['y']})
        return target_test_acc

