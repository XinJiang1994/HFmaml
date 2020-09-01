from sklearn.datasets import fetch_mldata
from tqdm import tqdm
import numpy as np
import random
import json
import os
import random
from tqdm import trange
import collections

random.seed(13)
np.random.seed(14)

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

    data_list = []
    label_list = []
    for i in tqdm(range(10)):
        idx = dataset['labels'] == i
        data_list.append(dataset['data'][idx])
        label_list.append(dataset['labels'][idx])
    return data_list,label_list


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
        self.dataset=self.get_final_dataset()
        self.train_data,self.test_data=self.generate_train_test()


    def get_pivots(self):
        accumulate_sum_r = [0]
        for i, r in enumerate(self.division_ratio):
            accumulate_sum_r.append(np.sum(self.division_ratio[:i + 1]))
        # user num is from 0 to 99,we use pivot_points to mark the bounds of different kinds of different nodes
        pivot_points = [int(self.num_users * r) for r in accumulate_sum_r]
        return pivot_points

    def dist1(self):#distribution 1
        #取前五类
        st_idx=self.pivots[0]
        end_idx=self.pivots[1]
        num_base=self.a
        dataset1 = {}
        sample_num=0
        for u in range(st_idx,end_idx):
            data_u=[]
            label_u=[]
            for class_idx in range(self.num_class//3):
                assert self.idx[class_idx]+num_base<self.data_list[class_idx].shape[0],'in dist1类别%d样本量不足'%class_idx
                data_u.append(self.data_list[class_idx][self.idx[class_idx]:self.idx[class_idx]+num_base])
                label_u.append(self.label_list[class_idx][self.idx[class_idx]:self.idx[class_idx]+num_base])
                self.idx[class_idx]+=num_base
                sample_num+=num_base
            data_u=np.concatenate(data_u)
            label_u=np.concatenate(label_u)
            dataset1[u]={'X':data_u,'y':label_u}
        ls = []
        for d_u in dataset1.values():
            # print('@line 182:',d_u)
            ls += d_u['y'].tolist()
        c = collections.Counter(ls)
        print('line 130 sample num dist1:', sample_num)
        print('@line 131 num of each class in dist1:', c)
        return dataset1

    def dist2(self): #ditribution2
        # 对每个user，先取a/2个前五类重的某一类，即，在0-4的类别中选择一类取a/2个样本
        # 然后在后五类中随机选一类取2a个样本
        st_idx = self.pivots[1]
        end_idx = self.pivots[2]
        dataset2={}
        sample_num=0
        # for u in range(st_idx, end_idx):
        #     data_u = []
        #     label_u = []
        #     for j in range(2):
        #         class_idx=(u+j)%self.num_class
        #         data_u.append(self.data_list[class_idx][self.idx[class_idx]:self.idx[class_idx] + self.a // 2])
        #         label_u.append(self.label_list[class_idx][self.idx[class_idx]:self.idx[class_idx]+self.a//2])
        #         self.idx[class_idx] += self.a // 2
        #     data_u = np.concatenate(data_u)
        #     label_u = np.concatenate(label_u)
        #     dataset2[u] = {'X': data_u, 'y': label_u}
        #     sample_num+=self.a//2

        for u in range(st_idx, end_idx):
            # 对每个user，先取a/2个前五类重的某一类，即，在0-4的类别中选择一类取a/2个样本
            class_idx=random.randint(0,self.num_class//2-1)
            assert self.idx[class_idx]+self.a<self.data_list[class_idx].shape[0],'in dist1类别%d样本量不足'%class_idx
            data_u = []
            label_u = []
            data_u.append(self.data_list[class_idx][self.idx[class_idx]:self.idx[class_idx]+self.a//2])
            label_u.append(self.label_list[class_idx][self.idx[class_idx]:self.idx[class_idx]+self.a//2])
            self.idx[class_idx]+=self.a//2
            #然后在后五类中随机选一类取2a个样本
            class_idx = random.randint(self.num_class//2, self.num_class-1)
            data_u.append(self.data_list[class_idx][self.idx[class_idx]:self.idx[class_idx] + self.a * 2])
            label_u.append(self.label_list[class_idx][self.idx[class_idx]:self.idx[class_idx] + self.a * 2])
            self.idx[class_idx] += self.a * 2
            data_u = np.concatenate(data_u)
            label_u = np.concatenate(label_u)
            dataset2[u]={'X':data_u,'y':label_u}
            sample_num+=self.a*2

        ls=[]
        for d_u in dataset2.values():
            # print('@line 182:',d_u)
            ls+=d_u['y'].tolist()
        c = collections.Counter(ls)
        print('line 223 sample num dist2:', sample_num)
        print('@line224 num of each class in dist2:', c)
        return dataset2

    def dist3(self):
        # 在前五类中随机采样
        st_idx = self.pivots[2]
        end_idx = self.pivots[3]
        dataset3 = {}
        sample_num=0
        for u in range(st_idx, end_idx):
            data_u = []
            label_u = []
            pi=np.random.rand(self.num_class)
            #print('@line 156 pi:',pi)
            # 随机丢弃一些类别
            # droplist=np.random.randint(2, size=self.num_class)
            # pi=pi*droplist
            pi[int(self.num_class//2):]=0
            # droplist=np.array([0]*self.num_class)
            r1=0
            r1 = random.randint(0, self.num_class//2 - 1)
            pi[r1]=0
            # r2=0
            # while r1==r2:
            #     r1=random.randint(0,self.num_class-1)
            #     r2=random.randint(0,self.num_class-1)
            # droplist[r1]=1
            # droplist[r2]=1
            # pi=pi*droplist

            for i in range(self.a*2):
                r=random.random()
                isSelect=pi>r
                for class_idx in range(self.num_class):
                    if isSelect[class_idx]:
                        data_u.append(self.data_list[class_idx][self.idx[class_idx]])
                        label_u.append(self.label_list[class_idx][self.idx[class_idx]])
                        self.idx[class_idx]+=1
                        sample_num+=1
            data_u = np.array(data_u)
            label_u = np.array(label_u) #因为每个元素只有一个，所以shape变成（）了,所以直接转array即可
            dataset3[u] = {'X': data_u, 'y': label_u}
        # print('@line 180 dataset3:',dataset3)
        ls=[]
        for d_u in dataset3.values():
            # print('@line 182:',d_u)
            ls+=d_u['y'].tolist()
        c = collections.Counter(ls)
        print('line 187 sample num dist3:', sample_num)
        print('@line184 num of each class in dist3:', c)
        return dataset3

    def dist4(self):
        # 在后五类中随机采样
        st_idx = self.pivots[3]
        end_idx = self.pivots[4]
        dataset4 = {}
        sample_num=0
        for u in range(st_idx, end_idx):
            data_u = []
            label_u = []
            pi=np.random.rand(self.num_class)
            #print('@line 156 pi:',pi)
            # 随机丢弃一些类别
            # droplist=np.random.randint(2, size=self.num_class)
            # pi=pi*droplist
            pi[:int(self.num_class//2)]=0
            # droplist = np.array([0] * self.num_class)
            # r1 = 0
            # r2 = 0
            # while r1 == r2:
            #     r1 = random.randint(0, self.num_class - 1)
            #     r2 = random.randint(0, self.num_class - 1)
            # droplist[r1] = 1
            # droplist[r2] = 1
            # pi = pi * droplist
            print('@line 206 pi:',pi)

            for i in range(self.a*2):
                r=random.random()
                isSelect=pi>r
                for class_idx in range(self.num_class):
                    if isSelect[class_idx]:
                        data_u.append(self.data_list[class_idx][self.idx[class_idx]])
                        label_u.append(self.label_list[class_idx][self.idx[class_idx]])
                        self.idx[class_idx]+=1
                        sample_num+=1
            data_u = np.array(data_u)
            label_u = np.array(label_u) #因为每个元素只有一个，所以shape变成（）了,所以直接转array即可
            dataset4[u] = {'X': data_u, 'y': label_u}
        ls=[]
        for d_u in dataset4.values():
            # print('@line 182:',d_u)
            ls+=d_u['y'].tolist()
        c = collections.Counter(ls)
        print('line 223 sample num dist4:', sample_num)
        print('@line224 num of each class in dist4:', c)
        return dataset4

    def dist5(self,n=2):
        # 循环取n类
        st_idx = self.pivots[4]
        end_idx = self.pivots[5]
        dataset = {}
        sample_num = 0
        c_pos=0
        for u in range(st_idx, end_idx):
            data_u = []
            label_u = []
            classes=[]
            c_st=c_pos
            c_end=c_pos+n

            for i in range(c_st,c_end):
                classes.append(i%self.num_class)
            print('@line 289 classes to get:',classes)
            c_pos += n

            for class_idx in classes:
                bias=random.randint(0,4)
                idx_st=self.idx[class_idx]
                idx_end=self.idx[class_idx]+self.a+bias
                data_u.append(self.data_list[class_idx][idx_st:idx_end])
                label_u.append(self.label_list[class_idx][idx_st:idx_end])
                self.idx[class_idx]+=self.a+bias
                sample_num+=self.a+bias
            data_u = np.concatenate(data_u)
            label_u = np.concatenate(label_u)
            dataset[u] = {'X': data_u, 'y': label_u}
        ls=[]
        for d_u in dataset.values():
            # print('@line 182:',d_u)
            ls+=d_u['y'].tolist()
        c = collections.Counter(ls)
        print('line 292 sample num dist5:', sample_num)
        print('@line293 num of each class in dist5:', c)
        return dataset

    def get_final_dataset(self):
        d1=self.dist1()
        d2=self.dist2()
        d3=self.dist3()
        d4=self.dist4()
        d5=self.dist5()

        d_fianl=d1
        d_fianl.update(d2)
        d_fianl.update(d3)
        d_fianl.update(d5)
        # d_fianl.update(d4)
        #打乱顺序
        uid=d_fianl.keys()
        udata=d_fianl.values()
        udata_new=[]
        for d in udata:
            X=d['X']
            y=d['y']
            # print("@line 156 X:", X)
            # print("@line 157 y:", y)
            y=y.reshape((-1,1))
            combined = np.concatenate((X,y),axis=1)
            # print(("@line 157 combined.shape:", combined.shape))
            np.random.shuffle(combined)
            split_pos=combined.shape[1]-1
            X,y=np.split(combined,[split_pos],axis=1)
            y=np.squeeze(y)
            udata_new.append({'X':X,'y':y})
            # print('@line 164 y',y)
            # print('@line 165 y.shape', y.shape)
            # print('test')
        random.shuffle(udata_new)
        d_fianl=dict(zip(uid,udata_new))
        d_fianl.update(d4)

        return d_fianl

    def solve_remained_data(self):
        data_r = []
        label_r = []
        for i in range(self.num_class):
            data_r.append(self.data_list[self.idx[i]:])
            label_r.append(self.label_list[self.idx[i]:])
        data_r = np.concatenate(data_r)
        label_r = np.concatenate(label_r)
        filename=os.path.join(self.savepath,'center_data.npy')
        np.savez(filename,X=data_r,y=label_r)

    def generate_train_test(self):
        train_data = {'users': [], 'user_data': {}, 'num_samples': []}
        test_data = {'users': [], 'user_data': {}, 'num_samples': []}
        # for i in range(self.num_users):
        #     uname = 'f_{0:05d}'.format(i)
        #     X=self.dataset[i]['X']
        #     y=self.dataset[i]['y']
        #     # train_len=int(self.train_test_ratio*X.shape[0])
        #     train_len = int(self.train_test_ratio * X.shape[0])
        #     test_len=X.shape[0]-train_len
        #     X=X.tolist()
        #     y=y.tolist()
        #     train_data['users'].append(uname)
        #     train_data['user_data'][uname] = {'x': X[:train_len], 'y': y[:train_len]}
        #     train_data['num_samples'].append(train_len)
        #     test_data['users'].append(uname)
        #     test_data['user_data'][uname] = {'x': X[train_len:], 'y': y[train_len:]}
        #     test_data['num_samples'].append(test_len)
        for i in range(self.num_users):
            uname = 'f_{0:05d}'.format(i)
            X = self.dataset[i]['X']
            y = self.dataset[i]['y']
            # train_len=int(self.train_test_ratio*X.shape[0])
            train_len = int(self.train_test_ratio * X.shape[0])
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

    def save_data(self):
        train_path = os.path.join(self.savepath, 'data/train/all_data_0_niid_0_keep_10_train_9.json')
        test_path = os.path.join(self.savepath, 'data/test/all_data_0_niid_0_keep_10_test_9.json')
        dir_path = os.path.dirname(train_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        dir_path = os.path.dirname(test_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(train_path, 'w') as outfile:
            json.dump(self.train_data, outfile)
        with open(test_path, 'w') as outfile:
            json.dump(self.test_data, outfile)



def genrate_cifar10():
    data_list, label_list=prepare_cifar10()
    generator=DataDivider(data_list,label_list,num_users=50,a=10,division_ratio=[0, 0, 0, 0, 1],train_test_ratio=0.2,savepath='./cifar10',num_class=10)
    generator.save_data()
    generator.solve_remained_data()

def genrate_mnist():
    # data_list, label_list=prepare_cifar10()
    data_list, label_list=prepare_mnist_data()
    generator=DataDivider(data_list,label_list,num_users=50,a=10,division_ratio=[0, 0, 0, 0, 1],train_test_ratio=0.2,savepath='./mnist',num_class=10)
    generator.save_data()
    generator.solve_remained_data()

def gennereate_dataset(dname='mnist'):
    if dname=='mnist':
        genrate_mnist()
    else:
        genrate_cifar10()

if __name__=='__main__':
    # gen_test()
    gennereate_dataset()