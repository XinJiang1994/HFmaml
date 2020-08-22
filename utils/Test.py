
#
# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict
# filenames=['batches.meta','data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']
# data_folder='/root/TC174611125/fmaml/HFmaml/data/cifar10/cifar10/'
# data_paths=[os.path.join(data_folder,f) for f in filenames]
#
# dataset={'data':[],'labels':[]}
# for p in data_paths[1:]:
#      cifars=unpickle(p)
#      data=cifars[b'data']
#      data=data.reshape(10000, 3, 32, 32)
#      labels=cifars[b'labels']
#      for i in range(data.shape[0]):
#          dataset['data'].append(np.transpose(data[i],[1,2,0]))
#          dataset['labels'].append(labels[i])
#
#
#
#
#
# # print(len(dataset['data']))
# # print(dataset['data'][0].shape)
#
# import matplotlib.pyplot as plt
# plt.imshow(dataset['data'][2],aspect="auto")
# plt.show()

from flearn.utils.model_utils import read_data
import os
# train_path = os.path.join('../data/cifar10', 'data', 'train')
# test_path = os.path.join('../data',  'cifar10', 'data', 'test')
# dataset = read_data(train_path, test_path)
#
# print(type(dataset[3]))
# print(dataset[3].keys())
# print(dataset[3]['f_00000'].keys())
# print('Training set')
# for i in range(10):
#     uname = 'f_{0:05d}'.format(i)
#     print(dataset[2][uname]['y'])
#
#
# print('testset')
# for i in range(10):
#     uname = 'f_{0:05d}'.format(i)
#     print(dataset[3][uname]['y'])


import tensorflow as tf

with  tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
    x=tf.get_variable(name='x',shape=(),initializer=tf.constant_initializer(9))
    print(x.shape)
    y=tf.get_variable(name='x',shape=())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y.load(3,sess)
    print(x,y)
    print('x',sess.run(x))
    print('y',sess.run(y))

##需要定义一个scope，然后全部是用tf.get_variable创建变量