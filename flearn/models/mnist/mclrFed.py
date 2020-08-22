import numpy as np
import tensorflow as tf
from flearn.models.FederateBaseModel import BaseModel

### This is an implenmentation of Hessian Free maml meta learning algirithm propoesed by Sheng Yue####

class Model(BaseModel):
    def __init__(self,params):
        self.num_classes=params['num_classes']
        super().__init__(params)

    def get_input(self):
        '''
        :return:the placeholders of input: features_train,labels_train,features_test,labels_test
        '''
        features_train = tf.placeholder(tf.float32, shape=[None, 784], name='features_train')
        labels_train = tf.placeholder(tf.float32, shape=[None, 10], name='labels_train')
        # features_test = tf.placeholder(tf.float32, shape=[None, 784], name='features_test')
        # labels_test = tf.placeholder(tf.float32, shape=[None, 10], name='labels_test')
        return features_train,labels_train

    def forward_func(self,inp, weights, w_names , reuse = False):

        '''
        :param inp: input
        :param weights: theta
        :param reuse:
        :return: model y
         when overload this function you should make w=dict(zip(w_names,weights))
        '''
        weights = dict(zip(w_names, weights))
        hidden = tf.matmul(inp, weights['w']) + weights['b']
        return hidden

    def construct_weights(self):
        '''
        :return:weights
        '''
        w = tf.Variable(tf.truncated_normal([784, self.num_classes], stddev=0.01), name='w')
        # weights['b1'] = tf.Variable(tf.zeros([self.num_classes]))
        b = tf.Variable(tf.zeros([self.num_classes]), name='b')
        return [w,b]

