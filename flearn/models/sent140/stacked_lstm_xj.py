import numpy as np
import tensorflow as tf
from flearn.models.BaseModel import BaseModel
import json
from flearn.utils.language_utils import line_to_indices
from flearn.utils.model_utils import batch_data_xin

with open('flearn/models/sent140/embs.json', 'r') as inf:
    embs = json.load(inf)
id2word = embs['vocab']
word2id = {v: k for k,v in enumerate(id2word)}
word_emb = np.array(embs['emba'])

def process_x(raw_x_batch, max_words=25):
    x_batch = [e[4] for e in raw_x_batch]
    x_batch = [line_to_indices(e, word2id, max_words) for e in x_batch]
    x_batch = np.array(x_batch)
    return x_batch

def process_y(raw_y_batch):
    y_batch = [1 if e=='4' else 0 for e in raw_y_batch]
    y_batch = np.array(y_batch)

    return y_batch

class Model(BaseModel):
    def __init__(self,model_params,alpha=0.01,rho=1.5,w_i=1,mu_i=0,seed=1):
        self.seq_len = model_params[0]
        self.num_classes = model_params[1]
        self.n_hidden = model_params[2]
        self.emb_arr = word_emb
        super().__init__(alpha,rho,w_i,mu_i,seed)

    def get_input(self):
        '''
        :return:the placeholders of input: features_train,labels_train,features_test,labels_test
        '''
        features = tf.placeholder(tf.int32, [None, self.seq_len], name='features')
        labels = tf.placeholder(tf.int64, [None, ], name='labels')
        return features,labels

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