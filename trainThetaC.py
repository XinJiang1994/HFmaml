import os

import tensorflow as tf
import tqdm
from tqdm import tqdm

# from flearn.utils.model_utils import active_func
from flearn.utils.model_utils import save_weights
from main_HFfmaml import reshape_features, reshape_label
from utils.model_utils import batch_data, read_data


def active_func(x, leak=0.2, name="active_func"):
    return tf.maximum(x, leak * x)
    # return tf.nn.elu(x)

def weight_variable(shape, name):
    # tf.set_random_seed(123)
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W, stride, padding='SAME'):
    return tf.nn.conv2d(x, W, strides=stride, padding=padding)

class Model():
    def __init__(self,params={}):
        self.channels=1
        self.stride=[1,1,1,1]
        self.training=True

    def get_input(self):
        '''
        :return:the placeholders of input: features_train,labels_train,features_test,labels_test
        '''
        features_train = tf.placeholder(tf.float32, shape=[None, 32,32,3], name='features_train')
        labels_train = tf.placeholder(tf.float32, shape=[None, 10], name='labels_train')
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

        h_conv3 = conv2d(h_pool2, weights['W_conv3'], stride=self.stride)
        h_conv3 = tf.nn.bias_add(h_conv3, weights['b_conv3'])
        h_conv3 = active_func(h_conv3)
        h_pool3 = tf.nn.avg_pool(h_conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME',
                                   name='avg_pool1')
        #
        # h_conv4 = conv2d(h_pool3, weights['W_conv4'], stride=self.stride)
        # h_conv4 = tf.nn.bias_add(h_conv4, weights['b_conv4'])
        # h_conv4 = active_func(h_conv4)
        # h_pool4 = tf.nn.avg_pool(h_conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME',
        #                       name='avg_pool1')
        # print(h_pool4)
        h_pool_shape=h_pool3.get_shape().as_list()
        h = h_pool_shape[1]
        w = h_pool_shape[2]
        c = h_pool_shape[3]
        flatten= tf.reshape(h_pool3, [-1, h * w * c])
        fc1 = tf.matmul(flatten, weights['W_fc1'])
        fc1 = tf.nn.bias_add(fc1, weights['b_fc1'])

        fc2 = tf.matmul(fc1, weights['W_fc2'])
        fc2 = tf.nn.bias_add(fc2, weights['b_fc2'])

        logits=tf.matmul(fc2, weights['W_fc3'])
        logits=tf.nn.bias_add(logits,weights['b_fc3'])
        return logits



    def construct_weights(self):
        '''
        :return:weights
        '''

        W_conv1 = weight_variable([3, 3, 3, 32], name='W_conv1')
        b_conv1 = bias_variable([32], name='b_conv1')

        W_conv2 = weight_variable([3, 3, 32, 64], name='W_conv2')
        b_conv2 = bias_variable([64], name='b_conv2')

        W_conv3 = weight_variable([3, 3, 64, 128], name='W_conv3')
        b_conv3 = bias_variable([128], name='b_conv3')
        #
        # W_conv4 = weight_variable([3, 3, 128, 256], name='W_conv4')
        # b_conv4 = bias_variable([256], name='b_conv4')

        W_fc1 = weight_variable([2048,512], name='W_fc1')
        b_fc1 = bias_variable([512], name='b_fc1')

        W_fc2 = weight_variable([512, 256], name='W_fc2')
        b_fc2 = bias_variable([256], name='b_fc2')

        W_fc3 = weight_variable([256, 10], name='W_fc3')
        b_fc3 = bias_variable([10], name='b_fc3')

        # weights=[W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_conv4,b_conv4,W_fc1,b_fc1]
        weights = [W_conv1, b_conv1, W_conv2, b_conv2,W_conv3,b_conv3, W_fc1, b_fc1,W_fc2,b_fc2,W_fc3,b_fc3]
        return weights

    def setTraining(self,isTraining):
        self.training=isTraining

    def train(self,train_data,test_data,bathsize,epoch=100):
        sess = tf.Session()
        features, labels=self.get_input()
        weights=self.construct_weights()
        w_names = [x.name.split(':', 1)[0] for x in weights]
        logits=self.forward_func(features,weights,w_names)
        pred=tf.argmax(logits,axis=1)
        acc=tf.reduce_mean(
            tf.cast(tf.equal(tf.cast(tf.argmax(input=labels, axis=1), dtype=tf.float32),
                             tf.cast(pred, dtype=tf.float32)), dtype=tf.float32))
        loss=tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss=tf.reduce_mean(loss)
        optimizer=tf.train.AdamOptimizer()
        g_and_v=optimizer.compute_gradients(loss)
        optimize_op = optimizer.apply_gradients(g_and_v, global_step=tf.train.get_global_step())
        sess.run(tf.global_variables_initializer())

        for i in tqdm(range(epoch)):
            for train_X, train_y in batch_data(train_data, bathsize):
                _,loss_val=sess.run([optimize_op,loss],feed_dict={features: train_X, labels: train_y})
            tqdm.write('round:{} loss:{}'.format(i,loss_val))
        test_batchsize=len(test_data['y'])

        print('train data num:',len(train_data['y']))
        print('test data num:',test_batchsize)
        for test_X,test_y in batch_data(test_data, test_batchsize):
            print(sess.run(acc,feed_dict={features: test_X, labels: test_y}))
        theta_c_path = '/root/TC174611125/fmaml/fmaml_mac/theta_c/{}_theata_c.mat'.format('cifar10')
        weights_val= sess.run(weights)
        save_weights(weights_val, w_names, theta_c_path)

def main():
    train_path = os.path.join('data', 'cifar10', 'data', 'pretrain')
    test_path = os.path.join('data', 'cifar10', 'data', 'pretest')
    dataset = read_data(train_path, test_path)
    num_class=10
    for user in dataset[0]:
        for i in range(len(dataset[2][user]['y'])):
            dataset[2][user]['x'][i] = reshape_features(dataset[2][user]['x'][i])
            dataset[2][user]['y'][i] = reshape_label(dataset[2][user]['y'][i], num_class)
    for user in dataset[0]:
        for i in range(len(dataset[3][user]['y'])):
            dataset[3][user]['x'][i] = reshape_features(dataset[3][user]['x'][i])
            dataset[3][user]['y'][i] = reshape_label(dataset[3][user]['y'][i], num_class)

    data_train_merge={'x':[],'y':[]}
    data_test_merge = {'x':[],'y':[]}
    for user in dataset[0]:
        data_train_merge['x']+=dataset[2][user]['x']
        data_train_merge['y']+=dataset[2][user]['y']
        data_test_merge['x']+=dataset[3][user]['x']
        data_test_merge['y']+=dataset[3][user]['y']
    # mv 8000 samples from testset to trainset
    testX1=data_test_merge['x'][:8000]
    testY1=data_test_merge['y'][:8000]
    testX2=data_test_merge['x'][8000:]
    testY2 = data_test_merge['y'][8000:]
    data_train_merge['x']+=testX1
    data_train_merge['y'] += testY1
    data_test_merge['x']=testX2
    data_test_merge['y'] = testY2
    model=Model()
    model.train(train_data=data_train_merge,test_data=data_test_merge,bathsize=20,epoch=50)


if __name__=='__main__':
    main()







