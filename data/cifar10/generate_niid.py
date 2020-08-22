from sklearn.datasets import fetch_mldata
from tqdm import tqdm
import numpy as np
import random
import json
import os
from tqdm import trange

# Setup directory for train/test data
train_path = './data/train/all_data_0_niid_0_keep_10_train_9.json'
test_path = './data/test/all_data_0_niid_0_keep_10_test_9.json'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

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

def prepare_mnist_data():
    # Get MNIST data, normalize, and divide by level
    mnist = fetch_mldata('MNIST original', data_home='./data')
    print(mnist.data.shape)
    mu = np.mean(mnist.data.astype(np.float32), 0)
    sigma = np.std(mnist.data.astype(np.float32), 0)
    mnist.data = (mnist.data.astype(np.float32) - mu)/(sigma+0.001)
    mnist_data = []
    mnist_label = []
    for i in tqdm(10):
        idx = mnist.target==i
        mnist_data.append(mnist.data[idx])
        mnist_label.append(mnist.target[idx])
    return mnist_data,mnist_label

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def prepare_cifar10():
    filenames = ['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5',
                 'test_batch']
    data_folder = '/root/TC174611125/fmaml/HFmaml/data/cifar10/cifar10/'
    data_paths = [os.path.join(data_folder, f) for f in filenames]
    dataset = {'data': [], 'labels': []}
    for p in data_paths[1:]:
        cifars = unpickle(p)
        data = cifars[b'data']
        # data = data.reshape(10000, 3, 32, 32)
        labels = cifars[b'labels']
        dataset['data'].append(data)
        dataset['labels'].append(labels)
    dataset['data']=np.concatenate(dataset['data'])
    dataset['labels']=np.concatenate(dataset['labels'])
    dataset['data']=norm(dataset['data'])

    print(dataset['labels'].shape)

    data_list = []
    label_list = []
    for i in tqdm(range(10)):
        idx = dataset['labels'] == i
        data_list.append(dataset['data'][idx])
        label_list.append(dataset['labels'][idx])
    return data_list,label_list






def generateNIID(data_list,label_list):
    '''
    data_list是一个list，里面保存的是每个类别的数据，例如mnist就是10个类别，data_list就存了10个元素，
    每个元素是一个ndarray,每个ndarray存的是一个类别的样本，是展开存储的，每一行是一个样本
    :param data_list: an ndarray
    :param label_list:
    :return:
    '''
    # print([len(v) for v in mnist_data])
    # print(mnist_label[2])

    ###### CREATE USER DATA SPLIT #######
    # Assign 10 samples to each user
    num_users=200

    X = [[] for _ in range(num_users)]
    y = [[] for _ in range(num_users)]
    idx = np.zeros(10, dtype=np.int64)
    for user in range(num_users):
        for j in range(5):
            l = (user+j)%10
            X[user] += data_list[l][idx[l]:idx[l]+5].tolist()
            #y[user] += (l*np.ones(5)).tolist()
            y[user] += label_list[l][idx[l]:idx[l]+5].tolist()
            idx[l] += 5
            #print(y[user])
    print(idx)

    # Assign remaining sample by power law
    user = 0
    props = np.random.lognormal(0, 0.2, (10,100,5)) #change 2.0 to 0.2
    props = np.array([[[len(v)-4500]] for v in data_list])*props/np.sum(props,(1,2), keepdims=True)#change 100 to 5000
    #idx = 1000*np.ones(10, dtype=np.int64)
    for user in trange(num_users):
        for j in range(5):
            l = (user+j)%10
            num_samples = int(props[l,user//10,j])
            #print(num_samples)
            if idx[l] + num_samples < len(data_list[l]):
                X[user] += data_list[l][idx[l]:idx[l]+num_samples].tolist()
                #y[user] += (l*np.ones(num_samples)).tolist()
                y[user] += label_list[l][idx[l]:idx[l]+num_samples].tolist()
                idx[l] += num_samples

    print(idx)

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    # Setup 100 users
    for i in trange(num_users, ncols=120):
        uname = 'f_{0:05d}'.format(i)

        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        #train_len = int(0.9*num_samples)
        train_len = 10
        test_len = num_samples - train_len

        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
        test_data['num_samples'].append(test_len)

    print(train_data['num_samples'])
    print(test_data['num_samples'])
    print(sum(train_data['num_samples']))
    print(sum(test_data['num_samples']))

    with open(train_path,'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)

datapath='./data/'
if not os.path.exists(datapath):
    os.mkdir(datapath)

def gen_test():
    filenames = ['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5',
                 'test_batch']
    data_folder = '/root/TC174611125/fmaml/HFmaml/data/cifar10/cifar10/'
    data_paths = [os.path.join(data_folder, f) for f in filenames]
    dataset = {'data': [], 'labels': []}
    for p in data_paths[1:]:
        cifars = unpickle(p)
        data = cifars[b'data']
        # data = data.reshape(10000, 3, 32, 32)
        labels = cifars[b'labels']
        dataset['data'].append(data)
        dataset['labels'].append(labels)
    dataset['data']=np.concatenate(dataset['data'])
    dataset['labels']=np.concatenate(dataset['labels'])
    dataset['data']=norm(dataset['data'])

    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}
    data_tmp=dataset['data']
    label_tmp=dataset['labels']
    for i in tqdm(range(100)):
        d=data_tmp[i:(i+1)*600]
        l=label_tmp[i:(i+1)*600]
        uname='f_{0:05d}'.format(i)
        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': d[:480], 'y': l[:480]}
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': d[480:], 'y': l[480:]}
        userdatapath=os.path.join(datapath,uname)
        if not os.path.exists(userdatapath):
            os.mkdir(userdatapath)
        trainX_fname=os.path.join(userdatapath,'trainX.npy')
        testX_fname=os.path.join(userdatapath,'testX.npy')

        trainY_fname=os.path.join(userdatapath,'trainY.npy')
        testY_fname=os.path.join(userdatapath,'testY.npy')

        np.save(trainX_fname,d[:480])
        np.save(trainY_fname,l[:480])
        np.save(testX_fname,d[:480])
        np.save(testY_fname,l[:480])

if __name__=='__main__':
    # gen_test()
    data_list,label_list=prepare_cifar10()

    generateNIID(data_list,label_list)