import numpy as np
import tensorflow as tf
from flearn.models.BaseModel import BaseModel

class Model(BaseModel):
    def __init__(self,params):
        self.num_classes=params['num_classes']
        super().__init__(params)

    def get_input(self):
        '''
        :return:the placeholders of input: features_train,labels_train,features_test,labels_test
        '''
        with self.graph.as_default():
            features_train = tf.placeholder(tf.float32, shape=[None, 60], name='features_train')
            labels_train = tf.placeholder(tf.float32, shape=[None, 10], name='labels_train')
            features_test = tf.placeholder(tf.float32, shape=[None, 60], name='features_test')
            labels_test = tf.placeholder(tf.float32, shape=[None, 10], name='labels_test')
        return features_train,labels_train,features_test,labels_test

    def forward_func(self,inp, weights, w_names , reuse = False):

        '''
        :param inp: input
        :param weights: theta
        :param reuse:
        :return: model y
         when overload this function you should make w=dict(zip(w_names,weights))
        '''
        with self.graph.as_default():
            weights = dict(zip(w_names, weights))
            hidden = tf.matmul(inp, weights['w']) + weights['b']
        return hidden

    def construct_weights(self):
        '''
        :return:weights
        '''
        with self.graph.as_default():
            w = tf.Variable(tf.truncated_normal([60, self.num_classes], stddev=0.01), name='w')
            b = tf.Variable(tf.zeros([self.num_classes]), name='b')
        # with self.graph.as_default():
        #     r = np.load('/root/TC174611125/fmaml/HFmaml/weights.npz')
        #     w = tf.Variable(r['w'], name='w')
        #     b = tf.Variable(r['b'], name='b')
        return [w,b]