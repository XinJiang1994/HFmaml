import numpy as np
import tensorflow as tf
from flearn.models.FederateBaseModel import BaseModel
from flearn.utils.model_utils import active_func


# def lrelu(x, leak=0.2, name="lrelu"):
#     return tf.maximum(x, leak * x)


def weight_variable(shape, name):
    # tf.set_random_seed(123)
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W, stride, padding='SAME'):
    return tf.nn.conv2d(x, W, strides=stride, padding=padding)

class Model(BaseModel):
    def __init__(self,params={}):
        self.num_classes=params['num_classes']
        self.channels=1
        self.stride=[1,1,1,1]
        self.training=True
        super(Model, self).__init__(params)

    def get_input(self):
        '''
        :return:the placeholders of input: features_train,labels_train,features_test,labels_test
        '''
        features = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='features_train')
        labels = tf.placeholder(tf.float32, shape=[None, 10], name='labels_train')

        features_test = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='features_test')
        labels_test = tf.placeholder(tf.float32, shape=[None, 10], name='labels_test')
        return features, labels, features_test, labels_test

    def forward_func(self,inp, weights, w_names , reuse = False):

        '''
        :param inp: input
        :param weights: theta
        :param reuse:
        :return: model y
         when overload this function you should make w=dict(zip(w_names,weights))
        '''
        weights = dict(zip(w_names, weights))

        h_conv1 = conv2d(inp, weights['W_conv1'], stride=[1, 1, 1, 1], padding='SAME')
        h_conv1 = tf.nn.bias_add(h_conv1, weights['b_conv1'])
        # h_conv1=tf.layers.batch_normalization(h_conv1,training=self.training)
        h_conv1 = active_func(h_conv1)
        h_pool1 = tf.nn.max_pool2d(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME',
                                   name='Max_pool1')

        h_conv2 = conv2d(h_pool1, weights['W_conv2'], stride=self.stride)
        h_conv2 = tf.nn.bias_add(h_conv2, weights['b_conv2'])
        h_conv2 = active_func(h_conv2)
        h_pool2 = tf.nn.max_pool2d(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME',
                                   name='Max_pool2')

        # h_conv3 = conv2d(h_pool2, weights['W_conv3'], stride=self.stride)
        # h_conv3 = tf.nn.bias_add(h_conv3, weights['b_conv3'])
        # h_conv3 = active_func(h_conv3)
        # h_pool3 = tf.nn.avg_pool(h_conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME',
        #                            name='avg_pool1')
        #
        # h_conv4 = conv2d(h_pool3, weights['W_conv4'], stride=self.stride)
        # h_conv4 = tf.nn.bias_add(h_conv4, weights['b_conv4'])
        # h_conv4 = active_func(h_conv4)
        # h_pool4 = tf.nn.avg_pool(h_conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME',
        #                       name='avg_pool1')
        # print(h_pool4)
        h_pool_shape=h_pool2.get_shape().as_list()
        h = h_pool_shape[1]
        w = h_pool_shape[2]
        c = h_pool_shape[3]
        flatten= tf.reshape(h_pool2, [-1, h * w * c])
        logits=tf.matmul(flatten, weights['W_fc1'])
        logits=tf.nn.bias_add(logits,weights['b_fc1'])

        #print(logits.shape)

        return logits



    def construct_weights(self):
        '''
        :return:weights
        '''

        W_conv1 = weight_variable([3, 3, 1, 32], name='W_conv1')
        b_conv1 = bias_variable([32], name='b_conv1')

        W_conv2 = weight_variable([3, 3, 32, 64], name='W_conv2')
        b_conv2 = bias_variable([64], name='b_conv2')

        # W_conv3 = weight_variable([3, 3, 64, 128], name='W_conv3')
        # b_conv3 = bias_variable([128], name='b_conv3')
        #
        # W_conv4 = weight_variable([3, 3, 128, 256], name='W_conv4')
        # b_conv4 = bias_variable([256], name='b_conv4')

        W_fc1 = weight_variable([3136,10], name='W_fc1')
        b_fc1 = bias_variable([10], name='b_fc1')

        # weights=[W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_conv4,b_conv4,W_fc1,b_fc1]
        weights = [W_conv1, b_conv1, W_conv2, b_conv2,
                   # W_conv3,b_conv3,
                   W_fc1, b_fc1]
        return weights

    def setTraining(self,isTraining):
        self.training=isTraining