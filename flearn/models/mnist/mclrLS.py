import numpy as np
import tensorflow as tf
from tqdm import trange
import operator

from tensorflow.contrib.layers.python import layers as tf_layers
#from flearn.utils.model_utils import batch_data
from flearn.utils.tf_utils import graph_size
from flearn.utils.tf_utils import process_grad
from flearn.utils.model_utils import batch_data_xin


class Model(object):
    """assumes that images are 28px by 28px"""

    def __init__(self, num_classes, opt1, opt2,train_data, test_data,rho=0.2,w_i=1,delta=0.1,mu_i=0.2, seed=1, num_local_updates=1):
        # we can define two optimizer here for two learning rate
        # optimizer is defined in fedavg,
        # learner is used in fedbase, learner is attribute of model
        # if we define model with two optimizer, learner has two
        # params, input learning rate, not optimizer
        #tf.global_variables_initializer().run()
        self.k=0
        self.num_classes = num_classes
        self.rho=rho
        self.delta=delta
        self.w_i=w_i
        self.mu_i=mu_i
        ## theta_k+1
        self.optimizer1 = tf.train.GradientDescentOptimizer(opt1)
        self.optimizer2 = tf.train.GradientDescentOptimizer(0)
        self.forward = self.forward_fc
        #init_op = tf.initialize_all_variables()
        # creat computation graph
        self.graph = tf.Graph()


        #self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123+seed)
            self.sess = tf.Session(graph=self.graph)
            self.yy_k = self.construct_yy_k()
            self.theta_kp1= self.construct_yy_k()
            #print(self.yy_k)
            self.sess.run(tf.global_variables_initializer())
            self.features_train, self.labels_train, self.features_test, self.labels_test, self.test_op,\
            self.grads, self.eval_metric_ops, self.loss, self.train_loss,self.theta_i_kp1, self.fast_vars = self.creat_model(opt1, num_local_updates)
            self.saver = tf.train.Saver()
            #self.sess.run(init_op)
            #self.sess = tf.Session(graph=self.graph)
        ## init y


        # find memory footprint and compute cost of the model
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops


    def creat_model(self, opt1,weights, num_local_updates=1):

        features_train = tf.placeholder(tf.float32, shape=[None, 784], name='features_train')
        labels_train = tf.placeholder(tf.float32, shape=[None, 10], name='labels_train')
        features_test = tf.placeholder(tf.float32, shape=[None, 784], name='features_test')
        labels_test = tf.placeholder(tf.float32, shape=[None, 10], name='labels_test')
        self.weights=weights = self.construct_fc_weights()
        logits_train = self.forward(features_train, weights, reuse = True)
        loss_train = self.xent(logits_train,labels_train)
        print("loss_train",loss_train)
        loss_train = tf.reduce_mean(loss_train)

        grads_and_vars = self.optimizer1.compute_gradients(loss_train)
        _,theta_kp1=zip(*grads_and_vars)
        theta_kp1=list(theta_kp1)
        phy=[]
        for grad,val in grads_and_vars:
            phy.append(val-opt1*grad)
        phy=dict(zip(weights.keys(),phy))

        logits_test = self.forward(inp=features_test,weights=phy, reuse=True)
        loss_test = self.xent(logits_test, labels_test)
        loss_test = tf.reduce_mean(loss_test)
        g_and_v = self.optimizer2.compute_gradients(loss_test)
        grad_phy,val_phy=zip(*g_and_v)
        #print(grad_phy)

        #prepare to compute theta_i_kp1
        tmp_sum_w=[]
        for th_kp1,gradphy in zip (theta_kp1,grad_phy):
            tmp_sum_w.append(th_kp1+self.delta*gradphy)
        tmp_sum_w=dict(zip(weights.keys(),tmp_sum_w))
        #print('theta_kp1',theta_kp1)
        logits_train_p = self.forward(features_train, tmp_sum_w, reuse=True)
        loss_train_p = self.xent(logits_train_p, labels_train)
        loss_train_p = tf.reduce_mean(loss_train_p)
        grads_and_vars_p = self.optimizer1.compute_gradients(loss_train_p)
        grad_1,_=zip(*grads_and_vars_p)

        tmp_min_w = []
        for th_kp1, gradphy in zip(theta_kp1, grad_phy):
            tmp_min_w.append(th_kp1 + self.delta * gradphy)
        tmp_min_w = dict(zip(weights.keys(), tmp_min_w))
        logits_train_m = self.forward(features_train, tmp_min_w, reuse=True)
        loss_train_m = self.xent(logits_train_m, labels_train)
        loss_train_m = tf.reduce_mean(loss_train_m)
        grads_and_vars_m = self.optimizer1.compute_gradients(loss_train_m)
        grad_2, _ = zip(*grads_and_vars_m)

        grad_1=list(grad_1)
        grad_2=list(grad_2)
        g_kp1=[]
        for g1,g2 in zip(grad_1,grad_2):
            g_kp1.append((g1-g2)/(2*self.delta))

        #print(type(self.yy_k))
        #print(type(grad_phy))
        theta_i_kp1s=[]
        for tpkp1,yy,g_phy,gg in zip(theta_kp1,self.yy_k,grad_phy,g_kp1):
            theta_i_kp1s.append(tpkp1-(yy+self.w_i*(g_phy-opt1*gg))/(self.rho+self.mu_i))

        th_i_kp1=dict(zip(weights.keys(), theta_i_kp1s))
        logits_test_final=self.forward(inp=features_test,weights=th_i_kp1, reuse=True)

        predictions_test = {
            "classes": tf.argmax(input=logits_test_final, axis=1),
            "probabilities": tf.nn.softmax(logits_test_final, name="softmax_tensor")
        }
        loss_test_final=self.xent(logits_test_final,labels_test)
        g_v=self.optimizer2.compute_gradients(loss_test_final)
        grads_test,_ = zip(*g_v)
        test_op = self.optimizer2.apply_gradients(g_v, global_step=tf.train.get_global_step())
        eval_metric_ops = tf.count_nonzero(tf.equal(tf.cast(tf.argmax(input=labels_test, axis=1),dtype=tf.float32), tf.cast(predictions_test["classes"],dtype=tf.float32)))
        # print('eval_metrics_ops shape: ',eval_metric_ops.shape)
        return features_train, labels_train, features_test, labels_test, test_op, grads_test, eval_metric_ops, loss_test, loss_train, theta_i_kp1s,phy



    def solve_inner(self, train_data, test_data, num_epochs):
        batch_size_train = len(train_data['y'])
        batch_size_test = len(test_data['y'])
        X_train, y_train= batch_data_xin(train_data,batch_size_train)
        X_test, y_test=batch_data_xin(test_data,batch_size_test)

        with self.graph.as_default():
            thikp1=self.sess.run(self.theta_i_kp1,
                                  feed_dict={self.features_train: X_train, self.labels_train: y_train,
                                             self.features_test: X_test, self.labels_test: y_test})
            self.set_params(thikp1)
            self.update_yy_k(thikp1)
        soln = thikp1
        yyk = self.get_yyk()
        comp = num_epochs * (len(test_data['y']) // batch_size_test) * batch_size_test * self.flops
        #comp = 0
        return soln, comp, yyk

    # def solve_inner(self, train_data, test_data, num_epochs):
    #     self.k+=1
    #     self.delta=1/(2**self.k)
    #     X_train=train_data['x']
    #     y_train=train_data['y']
    #     X_test=test_data['x']
    #     y_test=test_data['y']
    #     with self.graph.as_default():
    #         thikp1 = self.sess.run(self.theta_i_kp1,
    #                             feed_dict={self.features_train: X_train, self.labels_train: y_train,
    #                                         self.features_test: X_test, self.labels_test: y_test})
    #         self.update_yy_k(thikp1)
    #     # soln = self.get_theta_i_kp1()
    #     soln=thikp1
    #     yyk=self.get_yyk()
    #     #comp = num_epochs * (len(test_data['y']) // batch_size_test) * batch_size_test * self.flops
    #     comp=0
    #     return soln, comp,yyk

    def fast_adapt(self, train_data, num_epochs):
        batch_size_train = len(train_data['y'])
       #batch_size_test = len(test_data['y'])
        X_train, y_train = batch_data_xin(train_data, batch_size_train)
        #for X_test, y_test in batch_data(test_data, batch_size_test):
        with self.graph.as_default():
                soln=self.sess.run(self.fast_vars,
                            feed_dict={self.features_train: X_train, self.labels_train: y_train})
                                            # self.features_test: X_test, self.labels_test: y_test})
        #soln = self.get_params()
        comp = num_epochs * (len(train_data['y']) // batch_size_train) * batch_size_train * self.flops
        return soln, comp


    def set_params(self, model_params=None):
        self.theta_kp1 = model_params
        if model_params is not None:
            with self.graph.as_default():
                all_vars = tf.trainable_variables()
                tkp1s=[]
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)
                    tkp1s.append(variable)



    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    def get_theta_i_kp1(self):
        with self.graph.as_default():
            model_params = self.sess.run(self.fast_vars)
        return model_params

    def get_yyk(self):
        yy=[]
        for y in self.yy_k:
            with self.graph.as_default():
                y_v=self.sess.run(y)
                yy.append(y_v)
        #print(yy)
        return yy
    def set_yyk(self,yy):
        with self.graph.as_default():
            yyks=[]
            for y in yy:
                yyks.append(tf.convert_to_tensor(y))
            self.yy_k=yyks

    def update_yy_k(self,thikp1):
        with self.graph.as_default():
            yy_kp1=[]
            #print('tf.trainable_variables()',self.get_params())
            #print('self.theta_kp1',self.theta_kp1)
            ### thikp1 is theta_i_k+1
            ### tkp1s is theta_k+1
            #thikp1=self.get_theta_i_kp1()
            tkp1s=self.get_yyk()
            for yy_k,theta_kp1_i,theta_kp1 in zip(tkp1s,thikp1,self.theta_kp1):
                yy_kp1.append(tf.convert_to_tensor(yy_k+self.rho*(theta_kp1_i-theta_kp1)))
            self.yy_k=yy_kp1
        #print(self.yy_k)
        #print('test')

    def get_gradients(self, train_data, test_data, model_len):

        #grads_t = np.zeros(model_len)
        num_samples = len(test_data['y'])

        batch_size_train = len(train_data['y'])
        batch_size_test = len(test_data['y'])
        #for _ in trange(num_epochs=1, desc='Epoch: ', leave=False, ncols=120):
        X_train, y_train = batch_data_xin(train_data, batch_size_train)
        X_test, y_test = batch_data_xin(test_data, batch_size_test)
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
        weights['w1'] = tf.Variable(tf.truncated_normal([784, self.num_classes], stddev=0.01),name='w')
        #weights['b1'] = tf.Variable(tf.zeros([self.num_classes]))
        weights['b1'] = tf.Variable(tf.zeros([self.num_classes]),name='b')
        return weights
    def construct_yy_k(self):
        w= tf.truncated_normal([784, self.num_classes], stddev=0.01)
        #weights['b1'] = tf.Variable(tf.zeros([self.num_classes]))
        b= tf.zeros([self.num_classes])
        return [w,b]


    def test(self, train_data, test_data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''

        #print(train_data)
        batch_size_train =  len(train_data['y'])
        batch_size_test = len(test_data['y'])
        #tot_correct = []
        tot_correct = np.zeros(np.size(test_data['y']))
        loss = np.zeros(np.size(test_data['y']))
        # for _ in trange(num_epochs=1, desc='Epoch: ', leave=False, ncols=120):
        X_train,y_train,= batch_data_xin(train_data, batch_size_train)
        (X_test, y_test) = batch_data_xin(test_data, batch_size_test)
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
        X_test, y_test = batch_data_xin(test_data, batch_size_test)
        with self.graph.as_default():
            zero_loss =  self.sess.run(self.train_loss,
                              feed_dict={self.features_train: X_test, self.labels_train: y_test})

        return zero_loss


    def test_train(self, train_data):
        batch_size_train = len(train_data['y'])
        (X_train, y_train) = batch_data_xin(train_data, batch_size_train)
        with self.graph.as_default():
            self.sess.run(self.fast_vars,
                      feed_dict={self.features_train: X_train, self.labels_train: y_train})
        fast_vars=self.get_params()

        return fast_vars

# class Model(object):
#     """assumes that images are 28px by 28px"""
#
#     def __init__(self, num_classes, opt1, opt2, train_data, test_data, seed=1, num_local_updates=1):
#         # we can define two optimizer here for two learning rate
#         # optimizer is defined in fedavg,
#         # learner is used in fedbase, learner is attribute of model
#         # if we define model with two optimizer, learner has two
#         # params, input learning rate, not optimizer
#         #tf.global_variables_initializer().run()
#         self.num_classes = num_classes
#         self.construct_weights = self.construct_fc_weights
#         self.optimizer1 = tf.train.GradientDescentOptimizer(opt1)
#         self.optimizer2 = tf.train.GradientDescentOptimizer(opt2)
#         self.forward = self.forward_fc
#         #init_op = tf.initialize_all_variables()
#         # creat computation graph
#         self.graph = tf.Graph()
#
#
#         #self.graph = tf.Graph()
#         with self.graph.as_default():
#             tf.set_random_seed(123+seed)
#             self.sess = tf.Session(graph=self.graph)
#             self.sess.run(tf.global_variables_initializer())
#             self.features_train, self.labels_train, self.features_test, self.labels_test, self.test_op,\
#             self.grads, self.eval_metric_ops, self.loss, self.train_loss = self.creat_model(opt1, num_local_updates)
#             self.saver = tf.train.Saver()
#             #self.sess.run(init_op)
#             #self.sess = tf.Session(graph=self.graph)
#
#         # find memory footprint and compute cost of the model
#         self.size = graph_size(self.graph)
#         with self.graph.as_default():
#             self.sess.run(tf.global_variables_initializer())
#             metadata = tf.RunMetadata()
#             opts = tf.profiler.ProfileOptionBuilder.float_operation()
#             self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops
#
#
#     def creat_model(self, opt1, num_local_updates=1):
#
#         features_train = tf.placeholder(tf.float32, shape=[None, 784], name='features_train')
#         labels_train = tf.placeholder(tf.float32, shape=[None, 10], name='labels_train')
#         features_test = tf.placeholder(tf.float32, shape=[None, 784], name='features_test')
#         labels_test = tf.placeholder(tf.float32, shape=[None, 10], name='labels_test')
#
#
#         with tf.variable_scope('model', reuse=None) as training_scope:
#             if 'weights' in dir(self):
#                 training_scope.reuse_variables()
#                 weights = self.weights
#             else:
#                 # Define the weights
#                 self.weights = weights = self.construct_weights()
#
#
#         logits_train = self.forward(features_train, weights, reuse=True)
#         #logits_train = tf.layers.dense(inputs=features_train, units=self.num_classes,
#          #                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
#         predictions_train = {
#                 "classes": tf.argmax(input=logits_train, axis=1),
#                 "probabilities": tf.nn.softmax(logits_train, name="softmax_tensor")
#         }
#         #loss_train = tf.losses.sparse_softmax_cross_entropy(labels=labels_train, logits=logits_train)
#         loss_train = self.xent(labels_train, logits_train)
#         loss_train = tf.reduce_mean(loss_train)
#
#
#         grads = tf.gradients(loss_train, list(weights.values()))
#         gradients = dict(zip(weights.keys(), grads))
#         fast_vars = dict(zip(weights.keys(), [weights[key] - opt1*gradients[key] for key in weights.keys()]))
#
#         for j in range(num_local_updates-1):
#             print('current j: {}  and num_local_updates: {}'.format(j, num_local_updates))
#             logits_train = self.forward(features_train, fast_vars, reuse=True)
#             predictions_train = {
#                 "classes": tf.argmax(input=logits_train, axis=1),
#                 "probabilities": tf.nn.softmax(logits_train, name="softmax_tensor")
#             }
#             loss_train = self.xent(logits_train, labels_train)
#             loss_train = tf.reduce_mean(loss_train)
#             grads = tf.gradients(loss_train, list(fast_vars.values()))
#             gradients = dict(zip(fast_vars.keys(), grads))
#             fast_vars = dict(zip(fast_vars.keys(), [fast_vars[key] - opt1 * gradients[key] for key in fast_vars.keys()]))
#
#         #grads_and_vars = self.optimizer1.compute_gradients(loss_train)
#         #grads,vars = zip(*grads_and_vars)
#         #fast_vars = tuple(map(operator.add, vars, tuple((-opt1)*x for x in grads)))
#         #fast_vars = np.zeros(np.size(vars))
#         #for k in vars:
#         #fast_vars[0] = vars[0]-opt1*grads[0]
#         #gradients = dict(zip(vars.keys(), grads))
#         #fast_vars = dict(zip(vars.keys(), [vars[key] - opt1*gradients[key] for key in vars.keys()]))
#
#         #logits_test = tf.layers.dense(inputs=features_test, units=self.num_classes, kernel_initializer=fast_vars,
#         #                              kernel_regularizer=None)
#         logits_test = self.forward(features_test, fast_vars, reuse=True)
#         predictions_test = {
#             "classes": tf.argmax(input=logits_test, axis=1),
#             "probabilities": tf.nn.softmax(logits_test, name="softmax_tensor")
#         }
#         #loss_test = tf.losses.sparse_softmax_cross_entropy(labels=labels_test, logits=logits_test)
#         loss_test = self.xent(labels_test, logits_test)
#         loss_test = tf.reduce_mean(loss_test)
#
#
#         g_and_v = self.optimizer2.compute_gradients(loss_test)
#         grads_test,_ = zip(*g_and_v)
#         test_op = self.optimizer2.apply_gradients(g_and_v, global_step=tf.train.get_global_step())
#         #eval_metric_ops = tf.count_nonzero(tf.equal(labels_test, tf.cast(predictions_test["classes"], tf.float32)))
#         eval_metric_ops = tf.count_nonzero(tf.equal(tf.cast(tf.argmax(input=labels_test, axis=1), dtype=tf.float32),
#                                   tf.cast(predictions_test["classes"], dtype=tf.float32)))
#
#         return features_train, labels_train, features_test, labels_test, test_op, grads_test, eval_metric_ops, loss_test, loss_train
#
#
#
#
#
#
#     def solve_inner(self, train_data, test_data, num_epochs):
#         batch_size_train = len(train_data['y'])
#         batch_size_test = len(test_data['y'])
#         for _ in trange(num_epochs, desc='Epoch: ', leave=False, ncols=120):
#             for X_train, y_train in batch_data(train_data, batch_size_train):
#                 for X_test, y_test in batch_data(test_data, batch_size_test):
#                     with self.graph.as_default():
#                         self.sess.run(self.test_op,
#                                   feed_dict={self.features_train: X_train, self.labels_train: y_train,
#                                              self.features_test: X_test, self.labels_test: y_test})
#         soln = self.get_params()
#         comp = num_epochs * (len(test_data['y']) // batch_size_test) * batch_size_test * self.flops
#         return soln, comp
#
#
#
#
#
#     def set_params(self, model_params=None):
#         if model_params is not None:
#             with self.graph.as_default():
#                 all_vars = tf.trainable_variables()
#                 for variable, value in zip(all_vars, model_params):
#                     variable.load(value, self.sess)
#
#
#     def get_params(self):
#         with self.graph.as_default():
#             model_params = self.sess.run(tf.trainable_variables())
#         return model_params
#
#
#     def mse(self, label, pred):
#         label = tf.reshape(label, [-1])
#         pred = tf.reshape(pred, [-1])
#         return tf.reduce_mean(tf.sqrt(tf.square(pred-label)))
#
#
#     def xent(self, label, pred):
#         return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label)
#
#     def get_gradients(self, train_data, test_data, model_len):
#
#         #grads_t = np.zeros(model_len)
#         num_samples = len(test_data['y'])
#
#         batch_size_train = len(train_data['y'])
#         batch_size_test = len(test_data['y'])
#         #for _ in trange(num_epochs=1, desc='Epoch: ', leave=False, ncols=120):
#         for X_train, y_train in batch_data(train_data, batch_size_train):
#                 for X_test, y_test in batch_data(test_data, batch_size_test):
#                     with self.graph.as_default():
#                         model_grads = self.sess.run(self.grads,
#                                       feed_dict={self.features_train: X_train, self.labels_train: y_train,
#                                                  self.features_test: X_test, self.labels_test: y_test})
#
#         #with self.graph.as_default():
#         #    model_grads = self.sess.run(self.grads,
#         #                                feed_dict={self.features_test: data['x'], self.labels_test: data['y']})
#                         grads_t = process_grad(model_grads)
#
#         return num_samples, grads_t
#
#
#     def construct_fc_weights(self):
#         weights = {}
#         weights['w1'] = tf.Variable(tf.truncated_normal([784, self.num_classes], stddev=0.01))
#         weights['b1'] = tf.Variable(tf.zeros([self.num_classes]))
#        # weights['b1'] = tf.Variable(tf.zeros([self.num_classes]))
#         return weights
#
#    # def normalize(self, inp, activation, reuse, scope):
#         #if FLAGS.norm == 'batch_norm':
#         #    return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
#         #elif FLAGS.norm == 'layer_norm':
#     #    return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
#         #elif FLAGS.norm == 'None':
#         #    if activation is not None:
#         #        return activation(inp)
#         #    else:
#         #        return inp
#
#     def forward_fc(self, inp, weights, reuse=False):
#         dim_hidden = 1
#        # hidden = self.normalize(tf.matmul(inp, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse, scope='0')
#         hidden = tf.matmul(inp, weights['w1']) + weights['b1']
#         for i in range(1, 1):
#             hidden = tf.matmul(hidden, weights['w' + str(i + 1)]) #+ weights['b' + str(i + 1)]
#             #hidden = self.normalize(tf.matmul(hidden, weights['w' + str(i + 1)]) + weights['b' + str(i + 1)],
#             #                   activation=tf.nn.relu, reuse=reuse, scope=str(i + 1))
#         #return tf.matmul(hidden, weights['w'+str(1)]) + weights['b'+str(1)]
#         return hidden
#
#     def test(self, train_data, test_data):
#         '''
#         Args:
#             data: dict of the form {'x': [list], 'y': [list]}
#         '''
#         batch_size_train = len(train_data['y'])
#         batch_size_test = len(test_data['y'])
#         #tot_correct = []
#         tot_correct = np.zeros(np.size(test_data['y']))
#         loss = np.zeros(np.size(test_data['y']))
#         # for _ in trange(num_epochs=1, desc='Epoch: ', leave=False, ncols=120):
#         for X_train, y_train in batch_data(train_data, batch_size_train):
#             for X_test, y_test in batch_data(test_data, batch_size_test):
#                 with self.graph.as_default():
#                     tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
#                                               feed_dict={self.features_train: X_train, self.labels_train: y_train,
#                                                          self.features_test: X_test, self.labels_test: y_test})
#         return tot_correct, loss
#
#     def zeroth_loss(self, test_data):
#         batch_size_test = len(test_data['y'])
#         for X_test, y_test in batch_data(test_data, batch_size_test):
#             with self.graph.as_default():
#                 zero_loss = self.sess.run(self.train_loss,
#                                           feed_dict={self.features_train: X_test, self.labels_train: y_test})
#
#         return zero_loss
#
#
#     def close(self):
#         self.sess