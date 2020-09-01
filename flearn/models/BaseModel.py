import tensorflow as tf
import numpy as np
from flearn.utils.model_utils import batch_data_xin
from tensorflow.python import debug as tf_debug

### This is an implenmentation of Hessian Free maml meta learning algirithm propoesed by Sheng Yue####
### THis is Base model of the algorithm which implenments the optimization part####
###

class BaseModel(object):
    def __init__(self,params):
        # print('@BaseModel line 17 test init')
        self.k = 0
        self.alpha=params['alpha']
        self.rho = params['rho']
        self.w_i = params['w_i']
        self.mu_i = params['mu_i']
        self.seed=params['seed']
        self.graph = tf.Graph()
        self.theta_kp1 = None
        self.optimizer1 = tf.train.GradientDescentOptimizer(self.alpha)

        with self.graph.as_default():
            self.sess = tf.Session(graph=self.graph)
            #self.sess = tf_debug.TensorBoardDebugWrapperSession(self.sess, "TC174611125:32005")
            self.weights = self.construct_weights()  # weights is a list
            self.yy_k = self.construct_yy_k()
            tf.set_random_seed(123+self.seed)
            self.delta = tf.Variable(1000.0, dtype=tf.float32, trainable=False)
            self.features_train, self.labels_train, self.features_test, self.labels_test = self.get_input()
            self.eval_metric_ops, self.loss, self.train_loss, self.theta_i_kp1, self.fast_vars = self.optimize()
            self.saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())
            # self.summary_writer = tf.summary.FileWriter('./log/', self.sess.graph)
            # tf.summary.scalar("loss_train", self.train_loss)
            # tf.summary.scalar("loss_test(loss)", self.loss)
            # self.merged_summary_op = tf.summary.merge_all()

            #print('@BaseModel line 26 trainable variables',tf.trainable_variables())

    def optimize(self):
        with self.graph.as_default():
            w_names=[x.name.split(':',1)[0] for x in self.weights ]
            logits_train=self.forward_func(self.features_train,self.weights ,w_names,reuse=True)
            # print('baseModel line 44 logits_train.shape', logits_train.shape)
            loss_train=self.loss_func(logits_train,self.labels_train)

            # print('@BaseModel line 48',self.weights)

            grad_w=tf.gradients(loss_train,self.weights)
            # print('@BaseModel line 48 grad_w:',grad_w)
            phy=[val-self.alpha * grad for grad,val in zip(grad_w,self.weights)]

            g_v=self.optimizer1.compute_gradients(loss_train)
            self.adapt_op=self.optimizer1.apply_gradients(g_v, global_step=tf.train.get_global_step())

            logits_test=self.forward_func(inp=self.features_test,weights=phy,w_names=w_names, reuse=True)
            loss_test=self.loss_func(logits_test,self.labels_test)


            grad_Ltest2phy=tf.gradients(loss_test,phy)
            self.grad_Ltest2weight = tf.gradients(loss_test, self.weights)

            theta_kp1 = self.weights

            inner_g1=[th_kp1+self.delta*gradphy for th_kp1,gradphy in zip(theta_kp1,grad_Ltest2phy)]
            logits_train_p = self.forward_func(self.features_train, inner_g1,w_names=w_names, reuse=True)
            loss_train_p=self.loss_func(logits_train_p,self.labels_train)
            grad_1 = tf.gradients(loss_train_p,inner_g1)

            inner_g2 = [th_kp1 - self.delta * gradphy for th_kp1, gradphy in zip(theta_kp1, grad_Ltest2phy)]
            logits_train_m = self.forward_func(self.features_train, inner_g2,w_names=w_names, reuse=True)
            loss_train_m = self.loss_func(logits_train_m, self.labels_train)
            grad_2 = tf.gradients(loss_train_m, inner_g2)
            grad_1 = list(grad_1)
            grad_2 = list(grad_2)

            g_kp1 = [(g1 - g2) / (2 * self.delta) for g1, g2 in zip(grad_1, grad_2)]

            # hessian=list(tf.gradients(loss_test,self.weights))
            # grad_val=self.optimizer2.compute_gradients(loss_test,self.weights)

            theta_i_kp1s = [tpkp1-(yy+self.w_i*(g_phy-self.alpha*gg))/(self.rho+self.mu_i) for tpkp1,yy,g_phy,gg in zip(theta_kp1,self.yy_k,grad_Ltest2phy,g_kp1)]
            # theta_i_kp1s = [tpkp1 - (yy + self.w_i * (g_phy - 0* gg)) / (self.rho + self.mu_i) for
            #                 tpkp1, yy, g_phy, gg in zip(theta_kp1, self.yy_k, grad_Ltest2phy, g_kp1)]

            logits_test_final = self.forward_func(inp=self.features_test, weights=self.weights,w_names=w_names, reuse=True)

            self.predictions_test = {
                "classes": tf.argmax(input=logits_test_final, axis=1),
                "probabilities": tf.nn.softmax(logits_test_final, name="softmax_tensor")
            }
            pred_test=tf.argmax(input=logits_test, axis=1)
            self.test_acc=tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(input=self.labels_test, axis=1), dtype=tf.float32),
                                                        tf.cast(pred_test, dtype=tf.float32)),dtype=tf.float32))
            self.train_acc = tf.reduce_mean(
                tf.cast(tf.equal(tf.cast(tf.argmax(input=self.labels_train, axis=1), dtype=tf.float32),
                                 tf.cast(tf.argmax(input=logits_train,axis=1),dtype=tf.float32)), dtype=tf.float32))
            eval_metric_ops = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(input=self.labels_test, axis=1), dtype=tf.float32),
                                                        tf.cast(self.predictions_test["classes"], dtype=tf.float32)),dtype=tf.float32))
        return eval_metric_ops, loss_test, loss_train, theta_i_kp1s, phy

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

    def construct_yy_k(self):
        with self.graph.as_default():
            tv = tf.trainable_variables()
            yyk = [tf.Variable(tf.truncated_normal(x.shape, stddev=0.01),name='yyk_'+x.name.split(':',1)[0], dtype=tf.float32, trainable=False) for x in tv]
            # yyk = [tf.Variable(tf.zeros(x.shape), name='yyk_' + x.name.split(':', 1)[0],
            #                    dtype=tf.float32, trainable=False) for x in tv]
        return yyk

    def loss_func(self,logits,label):
        losses=tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label)
        return tf.reduce_mean(losses)

    def solve_inner(self, train_data, test_data, num_epochs):
        self.k += 1
        self.delta.load(1.0/(self.k+100)**2,self.sess)

        X_train=train_data['x']
        y_train=train_data['y']
        X_test=test_data['x']
        y_test=test_data['y']

        with self.graph.as_default():
            #print('@mclr lin 153: theta_kp1 before run', self.theta_kp1)
            thikp1=self.sess.run(self.theta_i_kp1,
                                  feed_dict={self.features_train: X_train, self.labels_train: y_train,
                                             self.features_test: X_test, self.labels_test: y_test})
            #print(thikp1)
            #self.update_yy_k(thikp1)
            # self.summary_writer.add_summary(summary, self.k)
        soln = thikp1
        # soln=self.get_params()
        yyk = self.get_yyk()
        return soln, yyk

    def fast_adapt(self, train_data, num_epochs):
        for i in range(num_epochs):
            with self.graph.as_default():
                self.sess.run(self.adapt_op,
                                feed_dict={self.features_train: train_data['x'], self.labels_train: train_data['y']})
                    # self.set_params(soln)
                soln=self.get_params()
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
                assert len(all_vars)==len(model_params),"set params error len(all_vars)!=len(model_params)"
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
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


    def test(self, train_data, test_data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''

        # print(train_data)

        with self.graph.as_default():
            acc_train,acc_test, loss, train_loss = self.sess.run([self.train_acc, self.test_acc, self.loss, self.train_loss],
                                                          feed_dict={self.features_train: train_data['x'],
                                                                     self.labels_train: train_data['y'],
                                                                     self.features_test: test_data['x'],
                                                                     self.labels_test: test_data['y']})
        return acc_train,acc_test, loss

    def test_test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        with self.graph.as_default():
            acc, loss_train,preds = self.sess.run([self.eval_metric_ops, self.train_loss,self.predictions_test['probabilities']],
                                                    feed_dict={self.features_train: data['x'], self.labels_train: data['y'],
                                                               self.features_test: data['x'], self.labels_test: data['y']})
        return acc, loss_train,preds
    def get_gradient_phy_w(self,train_data,test_data):
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

    def get_param_names(self):
        w_names = [x.name.split(':', 1)[0] for x in self.weights]
        return w_names

