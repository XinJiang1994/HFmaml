import tensorflow as tf
from flearn.utils.model_utils import batch_data

### This is an implenmentation of Hessian Free maml meta learning algirithm propoesed by Sheng Yue####
### THis is Base model of the algorithm which implenments the optimization part####
###

class BaseModel(object):
    def __init__(self,params):
        self.alpha=params['alpha']
        self.seed=params['seed']
        self.graph = tf.Graph()
        self.optimizer1 = tf.train.GradientDescentOptimizer(self.alpha)

        with self.graph.as_default():
            self.sess = tf.Session(graph=self.graph)
            #self.sess = tf_debug.TensorBoardDebugWrapperSession(self.sess, "TC174611125:32005")
            self.weights = self.construct_weights()  # weights is a list
            tf.set_random_seed(123+self.seed)
            self.features, self.labels,self.features_test,self.labels_test= self.get_input()
            self.eval_metric_ops,self.predictions_test, self.loss, self.optimize_op = self.optimize()
            self.saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())


    def optimize(self):
        with self.graph.as_default():
            w_names=[x.name.split(':',1)[0] for x in self.weights ]
            logits=self.forward_func(self.features,self.weights ,w_names,reuse=True)
            loss=self.loss_func(logits,self.labels)




            predictions_test = {
                "classes": tf.argmax(input=logits, axis=1),
                "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }
            g_and_v = self.optimizer1.compute_gradients(loss)
            fast_vars=[v - self.alpha * g for (g,v) in g_and_v]
            logits_test=self.forward_func(self.features_test, fast_vars ,w_names, reuse=True)
            self.loss_test=self.loss_func(logits_test,self.labels)
            pred_test = tf.argmax(input=logits_test, axis=1)
            self.test_acc = tf.reduce_mean(
                tf.cast(tf.equal(tf.cast(tf.argmax(input=self.labels_test, axis=1), dtype=tf.float32),
                                 tf.cast(pred_test, dtype=tf.float32)), dtype=tf.float32))

            grads_test, _ = zip(*g_and_v)
            optimize_op = self.optimizer1.apply_gradients(g_and_v, global_step=tf.train.get_global_step())
            eval_metric_ops = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(input=self.labels, axis=1), dtype=tf.float32),
                                                        tf.cast(predictions_test["classes"], dtype=tf.float32)),dtype=tf.float32))
        return eval_metric_ops,predictions_test, loss, optimize_op

    def get_input(self):
        '''
        :return:the placeholders of input: features_train,labels_train,features_test,labels_test
        '''
        pass

    def forward_func(self,inp, weights, w_names , reuse = False):

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


    def loss_func(self,logits,label):
        losses=tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label)
        return tf.reduce_mean(losses)

    def solve_inner(self, train_data, num_epochs):
        batch_size_train = len(train_data['y'])
        # batch_size_test = len(test_data['y'])
        for _ in range(num_epochs):
            for X_train, y_train in batch_data(train_data, batch_size_train):
                with self.graph.as_default():
                        self.sess.run(self.optimize_op ,
                                      feed_dict={self.features: X_train, self.labels: y_train})
        soln = self.get_params()
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


    def test(self, test_data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''

        # print(train_data)
        batch_size_test = len(test_data['y'])

        # X_train, y_train = batch_data(train_data, batch_size_train)
        # print('@FederateBaseModel line 133 ',test_data)

        with self.graph.as_default():
            acc, loss = self.sess.run([self.eval_metric_ops, self.loss],
                                                          feed_dict={self.features: test_data['x'],
                                                                     self.labels: test_data['y'],
                                                                    })
        return acc, loss

    def test_test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        with self.graph.as_default():
            acc, loss,preds = self.sess.run([self.eval_metric_ops, self.loss,self.predictions_test["classes"]],
                                                    feed_dict={self.features: data['x'], self.labels: data['y'],})
        return acc, loss,preds

    def fast_adapt(self, train_data, num_epochs):
        with self.graph.as_default():
            self.sess.run(self.optimize_op,
                        feed_dict={self.features: train_data['x'], self.labels: train_data['y']})
            soln=self.get_params()
        return soln


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

    def target_acc_while_train(self, train_data, test_data):
        target_test_acc = self.sess.run(self.test_acc,
                                            feed_dict={self.features: train_data['x'],
                                                       self.labels: train_data['y'],
                                                       self.features_test: test_data['x'],
                                                       self.labels_test: test_data['y']})
        return target_test_acc
        #with self.graph.as_default():
        #    model_grads = self.sess.run(self.grads,
        #                                feed_dict={self.features_test: data['x'], self.labels_test: data['y']})
        #                 grads_t = process_grad(model_grads)
