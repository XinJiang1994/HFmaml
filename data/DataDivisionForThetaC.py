from sklearn.datasets import fetch_mldata
from tqdm import tqdm
import numpy as np
import random
import json
import os
import random
from tqdm import trange
import collections
import struct

random.seed(13)
np.random.seed(14)
import argparse

class DataDivider():
    def __init__(self,data_list,label_list,num_users=100,a=100,division_ratio=[1/3,1/3,1/3],train_test_ratio=0.8,savepath = './cifar10/',num_class=10):
        assert np.sum(division_ratio) == 1, "sum of division_ratio not equal to 1"
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        self.data_list=data_list
        self.label_list=label_list
        self.num_users=num_users
        self.a=a
        self.division_ratio = division_ratio # ratio of users for each kind of distribution
        self.train_test_ratio=train_test_ratio # ratio of trainset and testset
        self.savepath=savepath
        self.num_class=num_class
        self.idx=[0]*num_class
        self.pivots=self.get_pivots()
        print('@line 95 pivots :',self.pivots)
        self.source_node,self.target_node,self.d_remain=self.get_final_dataset()
        self.source_train,self.source_test=self.generate_train_test(self.source_node,0.5)
        self.target_train,self.target_test=self.generate_train_test(self.target_node,3/10)
        self.train_data_remain, self.test_data_remain = self.generate_train_test(self.d_remain, 3 / 22)

        train_data = {'users': [], 'user_data': {}, 'num_samples': []}
        test_data = {'users': [], 'user_data': {}, 'num_samples': []}

        train_data['users']=self.source_train['users']+self.target_train['users']
        train_data['user_data']= {}
        train_data['user_data'].update(self.source_train['user_data'])
        train_data['user_data'].update(self.target_train['user_data'])
        train_data['num_samples']=self.source_train['num_samples']+self.target_train['num_samples']

        test_data['users'] = self.source_test['users'] + self.target_test['users']
        test_data['user_data'] = {}
        test_data['user_data'].update(self.source_test['user_data'])
        test_data['user_data'].update(self.target_test['user_data'])
        test_data['num_samples'] = self.source_test['num_samples'] + self.target_test['num_samples']
        self.train_data=train_data
        self.test_data=test_data

    def get_pivots(self):
        accumulate_sum_r = [0]
        for i, r in enumerate(self.division_ratio):
            accumulate_sum_r.append(np.sum(self.division_ratio[:i + 1]))
        # user num is from 0 to 99,we use pivot_points to mark the bounds of different kinds of different nodes
        pivot_points = [int(self.num_users * r) for r in accumulate_sum_r]
        return pivot_points

    def get_source_data(self,n=2,firstN_class=8):
        # 循环取n类
        st_idx = self.pivots[0]
        end_idx = self.pivots[1]
        dataset = {}
        sample_num = 0
        c_pos = 0
        for u in range(st_idx, end_idx):
            data_u = []
            label_u = []
            classes = []
            c_st = c_pos
            c_end = c_pos + n

            for i in range(c_st, c_end):
                classes.append(i % firstN_class) #从0-firstN_class类中取
            print('@line 289 classes to get:', classes)
            c_pos += n

            for class_idx in classes:
                random.seed(u)
                bias = random.randint(0, 10)
                idx_st = self.idx[class_idx]
                idx_end = self.idx[class_idx] + self.a + bias
                data_u.append(self.data_list[class_idx][idx_st:idx_end])
                label_u.append(self.label_list[class_idx][idx_st:idx_end])
                self.idx[class_idx] += self.a + bias
                sample_num += self.a + bias
            data_u = np.concatenate(data_u)
            label_u = np.concatenate(label_u)
            dataset[u] = {'X': data_u, 'y': label_u}
        ls = []
        for d_u in dataset.values():
            # print('@line 182:',d_u)
            ls += d_u['y'].tolist()
        c = collections.Counter(ls)
        print('line 292 sample num source:', sample_num)
        print('@line293 num of each class in source:', c)
        return dataset

    def get_target_data(self,n=2):
        # 循环取n类
        st_idx = self.pivots[1]
        end_idx = self.pivots[2]
        dataset = {}
        sample_num = 0
        c_pos = 0
        for u in range(st_idx, end_idx):
            data_u = []
            label_u = []
            classes = []
            c_st = c_pos
            c_end = c_pos + n

            for i in range(c_st, c_end):
                classes.append(i % (self.num_class)) #从10类中取
            print('@line 289 classes to get:', classes)
            c_pos += n

            for class_idx in classes:
                random.seed(u)
                bias = random.randint(10, 20)
                idx_st = self.idx[class_idx]
                idx_end = self.idx[class_idx] + self.a + bias
                data_u.append(self.data_list[class_idx][idx_st:idx_end])
                label_u.append(self.label_list[class_idx][idx_st:idx_end])
                self.idx[class_idx] += self.a + bias
                sample_num += self.a + bias
            data_u = np.concatenate(data_u)
            label_u = np.concatenate(label_u)
            dataset[u] = {'X': data_u, 'y': label_u}
        ls = []
        for d_u in dataset.values():
            # print('@line 182:',d_u)
            ls += d_u['y'].tolist()
        c = collections.Counter(ls)
        print('line 292 sample num source:', sample_num)
        print('@line293 num of each class in source:', c)
        return dataset

    def solve_remained_data(self):
        # cloud nodes
        n=2
        num_samples=self.a*5
        dataset = {}
        sample_num = 0
        c_pos = 0
        for u in range(self.num_users):
            data_u = []
            label_u = []
            classes = []
            c_st = c_pos
            c_end = c_pos + n
            for i in range(c_st, c_end):
                classes.append(i % (self.num_class-2)+2) # 从2-9类中取
            print('@line 289 classes to get:', classes)
            c_pos += n

            for class_idx in classes:
                random.seed(u)
                bias = random.randint(0, 10)
                idx_st = self.idx[class_idx]
                idx_end = self.idx[class_idx] + num_samples + bias
                data_u.append(self.data_list[class_idx][idx_st:idx_end])
                label_u.append(self.label_list[class_idx][idx_st:idx_end])
                self.idx[class_idx] += num_samples + bias
                sample_num += (num_samples + bias)
            data_u = np.concatenate(data_u)
            label_u = np.concatenate(label_u)
            dataset[u] = {'X': data_u, 'y': label_u}
        ls = []
        for d_u in dataset.values():
            # print('@line 182:',d_u)
            ls += d_u['y'].tolist()
        c = collections.Counter(ls)
        print('line 449 sample num dist5:', sample_num)
        print('@450 num of each class in solve_remained_data:', c)
        return dataset

    def get_final_dataset(self):
        source_node=self.get_source_data()
        taget_node=self.get_target_data()

        source_node=shuffle_data(source_node)
        taget_node=shuffle_data(taget_node)

        # d_fianl=source_node
        # d_fianl.update(taget_node)


        d_remain=self.solve_remained_data()
        d_remain=shuffle_data(d_remain)

        return source_node,taget_node,d_remain

    def generate_train_test(self,dataset,train_test_ratio):
        train_data = {'users': [], 'user_data': {}, 'num_samples': []}
        test_data = {'users': [], 'user_data': {}, 'num_samples': []}

        users=list(dataset.keys())
        for i in users:
            uname = 'f_{0:05d}'.format(i)
            X = dataset[i]['X']
            y = dataset[i]['y']
            # train_len=int(self.train_test_ratio*X.shape[0])
            train_len = int(train_test_ratio * X.shape[0])
            test_len = X.shape[0] - train_len
            X = X.tolist()
            y = y.tolist()
            train_data['users'].append(uname)
            train_data['user_data'][uname] = {'x': X[:train_len], 'y': y[:train_len]}
            train_data['num_samples'].append(train_len)
            test_data['users'].append(uname)
            test_data['user_data'][uname] = {'x': X[train_len:], 'y': y[train_len:]}
            test_data['num_samples'].append(test_len)
        print('num_samples of train data:',train_data['num_samples'])
        print('num_samples of test data:',test_data['num_samples'])
        print('sum num_samples of train data:',sum(train_data['num_samples']))
        print('sum num_samples of test data:',sum(test_data['num_samples']))
        return train_data,test_data

    def save_data(self,pretrain=False):

        if not pretrain:
            train_path = os.path.join(self.savepath,
                                          'data/train/all_data_train_u_{}_a{}_node_type{}.json'.format(self.num_users, self.a,
                                                                                                       5))
            test_path = os.path.join(self.savepath,
                                         'data/test/all_data_test_u{}_a{}_node_type{}.json'.format(self.num_users, self.a,
                                                                                                    5))
            # print(self.test_data)
            self.save(self.train_data,self.test_data,train_path,test_path)
        if pretrain:
            #solve the remain data
            train_path_r = os.path.join(self.savepath,
                                      'data/pretrain/remain_data_train_u_{}_a{}_node_type{}.json'.format(self.num_users, self.a,
                                                                                                   5))
            test_path_r = os.path.join(self.savepath,
                                     'data/pretest/remain_data_test_u{}_a{}_node_type{}.json'.format(self.num_users, self.a,
                                                                                                5))

            self.save(self.train_data_remain, self.test_data_remain, train_path_r, test_path_r)


    def save(self,train_data,test_data,train_path,test_path):
        # train_path = os.path.join(self.savepath, 'data/train/all_data_0_niid_0_keep_10_train_9.json')
        # test_path = os.path.join(self.savepath, 'data/test/all_data_0_niid_0_keep_10_test_9.json')
        # 如果之前有数据需要先删除
        os.system('rm -rf {}'.format(os.path.dirname(train_path)))
        print('rm -rf {}'.format(os.path.dirname(train_path)))
        os.system('rm -rf {}'.format(os.path.dirname(test_path)))
        print('rm -rf {}'.format(os.path.dirname(test_path)))
        dir_path = os.path.dirname(train_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        dir_path = os.path.dirname(test_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(train_path, 'w') as outfile:
            json.dump(train_data, outfile)
        with open(test_path, 'w') as outfile:
            json.dump(test_data, outfile)


def read_Fmnist_image(file_name):
    '''
    :param file_name: 文件路径
    :return:  训练或者测试数据
    如下是训练的图片的二进制格式
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    '''
    file_handle=open(file_name,"rb")  #以二进制打开文档
    file_content=file_handle.read()   #读取到缓冲区中
    head = struct.unpack_from('>IIII', file_content, 0)  # 取前4个整数，返回一个元组
    offset = struct.calcsize('>IIII')
    imgNum = head[1]  #图片数
    width = head[2]   #宽度
    height = head[3]  #高度
    bits = imgNum * width * height  # data一共有60000*28*28个像素值
    bitsString = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'
    imgs = struct.unpack_from(bitsString, file_content, offset)  # 取data数据，返回一个元组
    imgs_array=np.array(imgs).reshape((imgNum,width*height))     #最后将读取的数据reshape成 【图片数，图片像素】二维数组
    return imgs_array

def read_Fmnist_label(file_name):
    '''
    :param file_name:
    :return:
    标签的格式如下：
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.
    '''
    file_handle = open(file_name, "rb")  # 以二进制打开文档
    file_content = file_handle.read()  # 读取到缓冲区中
    head = struct.unpack_from('>II', file_content, 0)  # 取前2个整数，返回一个元组
    offset = struct.calcsize('>II')
    labelNum = head[1]  # label数
    bitsString = '>' + str(labelNum) + 'B'  # fmt格式：'>47040000B'
    label = struct.unpack_from(bitsString, file_content, offset)  # 取data数据，返回一个元组
    return np.array(label)
def get_data():
    # 文件获取
    train_image = "/root/TC174611125/Datasets/FashionMnist/train-images-idx3-ubyte"
    test_image = "/root/TC174611125/Datasets/FashionMnist/t10k-images-idx3-ubyte"
    train_label = "/root/TC174611125/Datasets/FashionMnist/train-labels-idx1-ubyte"
    test_label = "/root/TC174611125/Datasets/FashionMnist/t10k-labels-idx1-ubyte"
    # 读取数据
    train_x = read_Fmnist_image(train_image)
    test_x = read_Fmnist_image(test_image)
    train_y = read_Fmnist_label(train_label)
    test_y = read_Fmnist_label(test_label)
    return train_x,train_y,test_x,test_y

def prepare_Fmnist_data():
    train_x, train_y, test_x, test_y=get_data()
    data=np.concatenate([train_x,test_x])
    data=norm(data)
    labels=np.concatenate([train_y, test_y])
    data_list = []
    label_list = []
    for i in tqdm(range(10)):
        idx = labels == i
        data_list.append(data[idx])
        label_list.append(labels[idx])
    return data_list,label_list

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
    mnist = fetch_mldata('MNIST original', data_home='./mnist/data')
    print(mnist.data.shape)
    mu = np.mean(mnist.data.astype(np.float32), 0)
    sigma = np.std(mnist.data.astype(np.float32), 0)
    mnist.data = (mnist.data.astype(np.float32) - mu)/(sigma+0.001)
    mnist_data = []
    mnist_label = []
    for i in range(10):
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
    dataset['data']=np.concatenate(dataset['data'])
    dataset['labels']=np.concatenate(dataset['labels'])
    dataset['data']=norm(dataset['data'])

    print(dataset['labels'].shape)

    X = dataset['data']
    y = dataset['labels']
    # s_X=X.shape
    # s_y=y.shape
    # y=y.reshape((-1, 1))
    # combined = np.concatenate((X, y), axis=1)
    # np.random.seed(123)
    # np.random.shuffle(combined)
    # split_pos = s_X[1]
    # X, y = np.split(combined, [split_pos], axis=1)
    # # X=np.reshape(X,s_X)
    # y=np.squeeze(y)

    print('#####################y.shape:',y.shape)

    data_list = []
    label_list = []
    for i in tqdm(range(10)):
        idx = y == i
        data_list.append(X[idx])
        label_list.append(y[idx])
    return data_list,label_list

def prepare_cifar100():
    filenames = ['train','test']
    data_folder = '/root/TC174611125/Datasets/cifar100/'
    data_paths = [os.path.join(data_folder, f) for f in filenames]
    dataset = {'data': [], 'coarse_labels': [],'fine_labels':[]}
    for p in data_paths:
        cifars = unpickle(p)
        data = cifars[b'data']
        # data = data.reshape(10000, 3, 32, 32)
        coarse_labels = cifars[b'coarse_labels']
        fine_labels=cifars[b'fine_labels']
        dataset['data'].append(data)
        dataset['coarse_labels'].append(coarse_labels)
        dataset['fine_labels'].append(fine_labels)
    dataset['data']=np.concatenate(dataset['data'])
    dataset['coarse_labels']=np.concatenate(dataset['coarse_labels'])
    dataset['fine_labels'] = np.concatenate(dataset['fine_labels'])
    dataset['data']=norm(dataset['data'])

    data_list = []
    label_list = []
    for i in tqdm(range(100)):
        idx = dataset['fine_labels'] == i
        data_list.append(dataset['data'][idx])
        label_list.append(dataset['fine_labels'][idx])
    return data_list,label_list



def shuffle_data(d_fianl):
    # d_fianl 专用shuffle
    uid = d_fianl.keys()
    udata = d_fianl.values()
    udata_new = []
    for d in udata:
        X = d['X']
        y = d['y']
        # print("@line 156 X:", X)
        # print("@line 157 y:", y)
        y = y.reshape((-1, 1))
        combined = np.concatenate((X, y), axis=1)
        # print(("@line 157 combined.shape:", combined.shape))
        np.random.seed(123)
        np.random.shuffle(combined)
        split_pos = combined.shape[1] - 1
        X, y = np.split(combined, [split_pos], axis=1)
        y = np.squeeze(y)
        udata_new.append({'X': X, 'y': y})
        # print('@line 164 y',y)
        # print('@line 165 y.shape', y.shape)
        # print('test')
    random.shuffle(udata_new)
    d_fianl = dict(zip(uid, udata_new))
    return d_fianl


def genrate_cifar10(pretrain=False,user_num=50,a=10):
    data_list, label_list=prepare_cifar10()
    generator=DataDivider(data_list,label_list,num_users=user_num,a=a,division_ratio=[0.8,0.2],train_test_ratio=0.2,savepath='/root/TC174611125/fmaml/fmaml_mac/data/cifar10',num_class=10)
    generator.save_data(pretrain=pretrain)


if __name__=='__main__':
    genrate_cifar10(pretrain=True,user_num=100,a=10)
    genrate_cifar10(pretrain=False, user_num=50, a=10)