import json
import numpy as np
import os
from scipy import io
import numpy as np
import tensorflow as tf


def active_func(x):
    # return tf.maximum(x, leak * x)
    return tf.nn.elu(x)


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def load_weights(wPath='weights.mat'):
    params=io.loadmat(wPath)
    vars=list(params.values())[3:]
    # print(vars[0].shape)
    # print(vars[1].shape)
    # vars=[np.squeeze(x,axis=0) for x in vars]
    for i in range(len(vars)):
        if vars[i].shape[0]==1:
            vars[i]=np.squeeze(vars[i],axis=0)
    # print('----------------------------')
    # print(vars[2].shape)
    #print('@HFmaml line 85',vars)
    return vars

def save_weights(vars,names,savepath):
    vars=dict(zip(names,vars))
    io.savemat(savepath,vars)

def batch_data_xin(data,batch_size):
    data_x = data['x']
    data_y = data['y']

    # # randomly shuffle data
    # #np.random.seed(100)
    # #rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    # #np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    batched_x = data_x[0:batch_size]
    batched_y = data_y[0:batch_size]
    return (batched_x, batched_y)

def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)

def read_data_xin(datapath):
    clients = []
    groups = []
    train_data = {}
    test_data = {}
    unames=os.listdir(datapath)
    clients=unames
    for uname in unames:
        user_path=os.path.join(datapath,uname)
        trainX=np.load(os.path.join(user_path,'trainX.npy'))
        trainX=np.transpose(trainX.reshape(-1,3, 32, 32), [0, 2, 3, 1])
        trainY=np.load(os.path.join(user_path,'trainY.npy'))
        testX=np.load(os.path.join(user_path,'testX.npy'))
        testX = np.transpose(testX.reshape(-1,3, 32, 32), [0,2,3,1])
        testY=np.load(os.path.join(user_path,'testY.npy'))
        tmp_train={'x':trainX,'y':trainY.tolist()}
        tmp_test={'x':testX,'y':testY.tolist()}
        train_data[uname]=tmp_train
        test_data[uname]=tmp_test
    return clients, groups, train_data, test_data


def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data

class Metrics(object):
    def __init__(self, clients, params):
        self.params = params
        num_rounds = params['num_rounds']
        self.bytes_written = {c.id: [0] * num_rounds for c in clients}
        self.client_computations = {c.id: [0] * num_rounds for c in clients}
        self.bytes_read = {c.id: [0] * num_rounds for c in clients}      
        self.accuracies = []
        self.train_accuracies = []

    def update(self, rnd, cid, stats):
        bytes_w, comp, bytes_r = stats
        self.bytes_written[cid][rnd] += bytes_w
        self.client_computations[cid][rnd] += comp
        self.bytes_read[cid][rnd] += bytes_r

    def write(self):
        metrics = {}
        metrics['dataset'] = self.params['dataset']
        metrics['num_rounds'] = self.params['num_rounds']
        metrics['eval_every'] = self.params['eval_every']
        metrics['alpha'] = self.params['alpha']
        metrics['beta'] = self.params['beta']
        #metrics['mu'] = self.params['mu']
        metrics['num_epochs'] = self.params['num_epochs']
        metrics['batch_size'] = self.params['batch_size']
        metrics['accuracies'] = self.accuracies
        metrics['train_accuracies'] = self.train_accuracies
        metrics['client_computations'] = self.client_computations
        metrics['bytes_written'] = self.bytes_written
        metrics['bytes_read'] = self.bytes_read
        metrics_dir = os.path.join('/root/TC174611125/fmaml/out', self.params['dataset'], 'metrics_{}_{}_{}_{}_{}.json'.format(self.params['seed'], self.params['optimizer'], self.params['alpha'], self.params['beta'], self.params['num_epochs']))

        #safe_name = self.params['dataset'].replace('/', '_')

        if not os.path.exists(os.path.join('/root/TC174611125/fmaml/out', self.params['dataset'])):
            os.makedirs(os.path.join('/root/TC174611125/fmaml/out', self.params['dataset']))
        with open(metrics_dir, 'w') as ouf:
            json.dump(metrics, ouf)
