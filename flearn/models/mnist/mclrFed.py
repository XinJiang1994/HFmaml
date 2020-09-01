import numpy as np
import tensorflow as tf
from flearn.models.FederateBaseModel import BaseModel

### This is an implenmentation of Hessian Free maml meta learning algirithm propoesed by Sheng Yue####
from flearn.utils.model_utils import lrelu


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
        weights = dict(zip(w_names, weights))
        hidden = tf.matmul(inp, weights['w1']) + weights['b1']
        hidden = lrelu(hidden)
        hidden = tf.matmul(hidden, weights['w2']) + weights['b2']
        hidden = lrelu(hidden)
        hidden = tf.matmul(hidden, weights['w3']) + weights['b3']
        return hidden

    def construct_weights(self):
        '''
        :return:weights
        '''
        w1 = tf.Variable(tf.truncated_normal([784, 1024], stddev=0.01), name='w1')
        b1 = tf.Variable(tf.zeros([1024]), name='b1')
        w2 = tf.Variable(tf.truncated_normal([1024, 512], stddev=0.01), name='w2')
        b2 = tf.Variable(tf.zeros([512]), name='b2')
        w3 = tf.Variable(tf.truncated_normal([512, self.num_classes], stddev=0.01), name='w3')
        b3 = tf.Variable(tf.zeros([self.num_classes]), name='b3')
        return [w1, b1, w2, b2, w3, b3]
