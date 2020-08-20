import numpy as np
import tensorflow as tf

from tensorflow.contrib.layers.python import layers as tf_layers
from flearn.utils.tf_utils import graph_size
from flearn.utils.model_utils import batch_data_xin


class Model(object):
    """assumes that images are 28px by 28px"""

    def __init__(self, num_classes, opt1, rho=0.2,w_i=1,mu_i=0.2, seed=1):
        # we can define two optimizer here for two learning rate
        # optimizer is defined in fedavg,
        # learner is used in fedbase, learner is attribute of model
        # if we define model with two optimizer, learner has two
        # params, input learning rate, not optimizer
        #tf.global_variables_initializer().run()
        self.k=0
        self.num_classes = num_classes
        self.rho=rho
        self.w_i=w_i
        self.mu_i=mu_i
        self.forward = self.forward_fc
        self.graph = tf.Graph()
        #self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123+seed)
            self.sess = tf.Session(graph=self.graph)
            self.delta = tf.Variable(1, dtype=tf.float32, trainable=False)
            self.yy_k = self.construct_yy_k()
            self.theta_kp1= None#这个初始化没有，因为platform会传值给它
            self.features_train, self.labels_train, self.features_test, self.labels_test, \
            self.eval_metric_ops, self.loss, self.train_loss,self.theta_i_kp1, self.fast_vars = self.creat_model(opt1)
            self.saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())
        # find memory footprint and compute cost of the model
        self.size = graph_size(self.graph)

    def creat_model(self, opt1):

        features_train = tf.placeholder(tf.float32, shape=[None, 784], name='features_train')
        labels_train = tf.placeholder(tf.float32, shape=[None, 10], name='labels_train')
        features_test = tf.placeholder(tf.float32, shape=[None, 784], name='features_test')
        labels_test = tf.placeholder(tf.float32, shape=[None, 10], name='labels_test')
        self.weights = self.construct_fc_weights()
        logits_train = self.forward(features_train, self.weights, reuse = True)
        loss_train = self.xent(logits_train,labels_train)
        #print("loss_train",loss_train)
        loss_train = tf.reduce_mean(loss_train)

        grads_v = tf.gradients(loss_train,list(self.weights.values()))
        #grad_v is a list which contains grads of w and b
        grads_and_vars=zip(grads_v,list(self.weights.values()))
        phy=[]
        for grad,val in grads_and_vars:
            phy.append(val-opt1*grad)
        phy=dict(zip(self.weights.keys(),phy))

        logits_test = self.forward(inp=features_test,weights=phy, reuse=True)
        loss_test = self.xent(logits_test, labels_test)
        loss_test = tf.reduce_mean(loss_test)
        grad_phy = tf.gradients(loss_test,list(phy.values()))

        #prepare to compute theta_i_kp1
        # 每轮训练前server 通过set_param方法将theta_k+1传过来（更新w）所以这里的weights就是theta_k+1,如果是mini-batch每次多轮的话就不一样了
        theta_kp1 = list(self.weights.values())
        inner_g1=[]
        for th_kp1,gradphy in zip (theta_kp1,grad_phy):
            inner_g1.append(th_kp1+self.delta*gradphy)
        inner_g1=dict(zip(self.weights.keys(),inner_g1))
        logits_train_p = self.forward(features_train, inner_g1, reuse=True)
        loss_train_p = self.xent(logits_train_p, labels_train)
        loss_train_p = tf.reduce_mean(loss_train_p)
        ## inner_g grad
        grad_1 = tf.gradients(loss_train_p,list(inner_g1.values()))

        inner_g2 = []
        for th_kp1, gradphy in zip(theta_kp1, grad_phy):
            inner_g2.append(th_kp1 - self.delta * gradphy)
        inner_g2 = dict(zip(self.weights.keys(), inner_g2))
        logits_train_m = self.forward(features_train, inner_g2, reuse=True)
        loss_train_m = self.xent(logits_train_m, labels_train)
        loss_train_m = tf.reduce_mean(loss_train_m)
        grad_2 = tf.gradients(loss_train_m,list(inner_g2.values()))

        grad_1=list(grad_1)
        grad_2=list(grad_2)
        g_kp1=[]
        for g1,g2 in zip(grad_1,grad_2):
            g_kp1.append((g1-g2)/(2*self.delta))
        theta_i_kp1s=[]
        for tpkp1,yy,g_phy,gg in zip(theta_kp1,self.yy_k,grad_phy,g_kp1):
            theta_i_kp1s.append(tpkp1-(yy+self.w_i*(g_phy-opt1*gg))/(self.rho+self.mu_i))

        logits_test_final=self.forward(inp=features_test,weights=self.weights, reuse=True)

        self.predictions_test = {
            "classes": tf.argmax(input=logits_test_final, axis=1),
            "probabilities": tf.nn.softmax(logits_test_final, name="softmax_tensor")
        }
        eval_metric_ops = tf.count_nonzero(tf.equal(tf.cast(tf.argmax(input=labels_test, axis=1),dtype=tf.float32), tf.cast(self.predictions_test["classes"],dtype=tf.float32)))
        return features_train, labels_train, features_test, labels_test, eval_metric_ops, loss_test, loss_train, theta_i_kp1s,phy



    def solve_inner(self, train_data, test_data, num_epochs):
        batch_size_train = len(train_data['y'])
        batch_size_test = len(test_data['y'])
        self.k += 1
        self.delta.load(1/(self.k+10)**2,self.sess)
        X_train, y_train= batch_data_xin(train_data,batch_size_train)
        X_test, y_test=batch_data_xin(test_data,batch_size_test)

        with self.graph.as_default():
            #print('@mclr lin 153: theta_kp1 before run', self.theta_kp1)
            thikp1=self.sess.run(self.theta_i_kp1,
                                  feed_dict={self.features_train: X_train, self.labels_train: y_train,
                                             self.features_test: X_test, self.labels_test: y_test})
            self.update_yy_k(thikp1)

        soln = thikp1
        yyk = self.get_yyk()
        # comp = num_epochs * (len(test_data['y']) // batch_size_test) * batch_size_test * self.flops
        # comp = 0
        return soln, yyk

    def fast_adapt(self, train_data, num_epochs):
        batch_size_train = len(train_data['y'])
        for i in range(num_epochs):
            X_train, y_train = batch_data_xin(train_data, batch_size_train)
            with self.graph.as_default():
                    soln=self.sess.run(self.fast_vars,
                                feed_dict={self.features_train: X_train, self.labels_train: y_train})
                    self.set_params(list(soln.values()))
            #soln = self.get_params()
            #comp = num_epochs * (len(train_data['y']) // batch_size_train) * batch_size_train * self.flops
            comp=0
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
                all_vars = tf.trainable_variables()
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    def get_theta_i_kp1(self):
        with self.graph.as_default():
            model_params = self.sess.run(self.theta_i_kp1)
        return model_params

    def get_yyk(self):
        with self.graph.as_default():
            yy=self.sess.run(self.yy_k)
        return yy
    def set_yyk(self,vals):
        with self.graph.as_default():
            for y,v in zip(self.yy_k,vals):
                y.load(v,self.sess)

    def update_yy_k(self,thikp1):
        with self.graph.as_default():
            yy_kp1s=[]
            yy_ks=self.get_yyk()
            for yy_k,theta_kp1_i,theta_kp1 in zip(yy_ks,thikp1,self.theta_kp1):
                yy_kp1s.append(yy_k+self.rho*(theta_kp1_i-theta_kp1))
            self.set_yyk(yy_kp1s)


    def construct_fc_weights(self):
        weights = {}
        weights['w1'] = tf.Variable(tf.truncated_normal([784, self.num_classes], stddev=0.01),name='w')
        #weights['b1'] = tf.Variable(tf.zeros([self.num_classes]))
        weights['b1'] = tf.Variable(tf.zeros([self.num_classes]),name='b')
        return weights
    def construct_yy_k(self):
        w = tf.Variable(tf.truncated_normal([784, self.num_classes], stddev=0.01),name='yyw',trainable=False)
        b = tf.Variable(tf.zeros([self.num_classes]),name='yyb',trainable=False)
        # w = tf.Variable(tf.zeros([784, self.num_classes]), name='yyw', trainable=False)
        # b = tf.Variable(tf.zeros([self.num_classes]), name='yyb', trainable=False)
        return [w,b]


    def test(self, train_data, test_data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''

        #print(train_data)
        batch_size_train =  len(train_data['y'])
        batch_size_test = len(test_data['y'])

        X_train,y_train= batch_data_xin(train_data, batch_size_train)
        X_test, y_test = batch_data_xin(test_data, batch_size_test)
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
            tot_correct, loss_train,preds = self.sess.run([self.eval_metric_ops, self.train_loss,self.predictions_test["classes"]],
                                                    feed_dict={self.features_train: data['x'], self.labels_train: data['y'],
                                                               self.features_test: data['x'], self.labels_test: data['y']})
        return tot_correct, loss_train,preds


    def close(self):
        self.sess

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
