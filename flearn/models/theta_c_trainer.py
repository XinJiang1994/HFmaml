import os
import numpy as np
import tensorflow as tf
from flearn.utils.model_utils import batch_data,save_weights
from tqdm import tqdm

def norm(data):
    '''
    data is an ndarray with shape N,w,h,c
    :param data:
    :return:
    '''
    mu = np.mean(data.astype(np.float32), 0)
    sigma = np.std(data.astype(np.float32), 0)
    data_norm= (data.astype(np.float32) - mu) / (sigma + 0.001)
    return data_norm

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def reshape_label(label,n=10):
    assert len(label.shape)==1,'Reshape labe error, Label shape is not (n,)'
    print('@theta_c_trainer line25 lable.shape:',label.shape)
    label_new=[]
    for l in label:
        newL=[0]*n
        newL[l]=1
        label_new.append(newL)
    return np.array(label_new)


def reshape_features(x):
    x = np.array(x)
    x = np.transpose(x.reshape(3, 32, 32), [1, 2, 0])
    # print(x.shape)
    return x

class ThetaCModelBase():
    def __init__(self,model_name='mclr2'):
        self.model_name=model_name
        self.data_path=''
        self.thetaC_savepath=''
        self.setting()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session(graph=self.graph)
            #self.sess = tf_debug.TensorBoardDebugWrapperSession(self.sess, "TC174611125:32005")
            tf.set_random_seed(123)
            self.weights = self.construct_weights()  # weights is a list
            # print('@theta_c_trainer line 19:\n',self.weights)
            self.features, self.labels = self.get_input()
            self.build_model()

    def setting(self):
        self.thetaC_savepath = '/root/TC174611125/fmaml/fmaml_mac/theta_c/{}_theata_c.mat'.format(self.model_name)
        dir_path = os.path.dirname(self.thetaC_savepath)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if self.model_name=='cifar10':
            self.data_path='/root/TC174611125/fmaml/fmaml_mac/data/cifar10/center_data.npz'
        else:
            self.data_path = '/root/TC174611125/fmaml/fmaml_mac/data/mnist/center_data.npz'

    def build_model(self):
        w_names=[x.name.split(':',1)[0] for x in self.weights ]
        self.logits=self.forward_func(self.features,self.weights ,w_names,reuse=True)
        self.loss=self.loss_func(self.logits,self.labels)
        vars=tf.trainable_variables()
        self.optmizer = tf.train.AdamOptimizer(0.01, 0.999).minimize(self.loss, var_list=vars)

        self.pred=tf.argmax(input=self.logits, axis=1)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.cast(tf.argmax(input=self.labels, axis=1), dtype=tf.float32),
                                 tf.cast(self.pred, dtype=tf.float32)), dtype=tf.float32))
        self.sess.run(tf.global_variables_initializer())

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

    def load_data(self):
        filenames = ['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5',
                     'test_batch']
        data_folder = '/root/TC174611125/Datasets/cifar10/'
        data_paths = [os.path.join(data_folder, f) for f in filenames]
        dataset = {'data': [], 'labels': []}
        for p in data_paths[1:]:
            cifars = unpickle(p)
            data = cifars[b'data']
            # data = data.reshape(10000, 3, 32, 32)
            labels = cifars[b'labels']
            dataset['data'].append(data)
            dataset['labels'].append(labels)
        dataset['data'] = np.concatenate(dataset['data'])
        dataset['labels'] = np.concatenate(dataset['labels'])
        dataset['data'] = norm(dataset['data'])
        dataset['data']=dataset['data'].reshape([-1,3,32,32])
        dataset['data']=dataset['data'].transpose([0,2,3,1])
        dataset['labels']=reshape_label(dataset['labels'])

        return {'x':dataset['data'] ,'y':dataset['labels']}

    def train(self, bath_size,num_epoch):
        data = self.load_data()
        # vars = tf.trainable_variables()

        for i in tqdm(range(num_epoch)):
            for X, y in batch_data(data, bath_size):
                # print('@theta_c_trainer line 83 X.shape:',X.shape)
                self.sess.run([self.optmizer],feed_dict={self.features:X,self.labels:y})
        acc=self.test()
        print('Accuracy:',acc)
    def get_params(self):
        params=self.sess.run(self.weights)
        # print('params:',params)
        return params

    def get_param_names(self):
        w_names = [x.name.split(':', 1)[0] for x in self.weights]
        return w_names

    def save_thetaC(self):
        params=self.get_params()
        w_names=self.get_param_names()
        save_weights(params, w_names, self.thetaC_savepath)
    def test(self):
        data = self.load_data()
        acc=self.sess.run(self.accuracy, feed_dict={self.features: data['x'], self.labels: data['y']})
        return acc
















