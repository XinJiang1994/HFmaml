from scipy import io
import numpy as np
from flearn.models.ThetaC.cnn_model import Model
from flearn.utils.model_utils import load_weights

# def load_weights(wPath='weights.mat'):
#     params=io.loadmat(wPath)
#     vars=list(params.values())[3:]
#     vars=[np.squeeze(x) for x in vars]
#     #print('@HFmaml line 85',vars)
#     return vars
#
# params=io.loadmat('weights.mat')
#
# print(params)
#
# import tensorflow as tf
#
# x=tf.convert_to_tensor(np.array([[-1.0,-10,-2.5,0.0,-1.0,-5.1,-5.1,-45,-1.2,-3.5], [1.0,-10,2.5,0.0,1.0,5.1,1.3,-145,1.2,3.5]]))
# y=tf.nn.softmax(x)
# pred=tf.argmax(y,axis=1)
# pred2=tf.argmax(x,axis=1)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(y))
#     print(sess.run(pred))
#     print(sess.run(pred2))
# import os
#
# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict
#
# filenames = ['train','test']
# data_folder = '/root/TC174611125/Datasets/cifar100/'
# data_paths = [os.path.join(data_folder, f) for f in filenames]
#
# for p in data_paths:
#     data=unpickle(p)
#     print(data.keys())


def main():
    theta_c = load_weights('/root/TC174611125/fmaml/fmaml_mac/theta_c/cifar10_theata_c.mat')
    print(theta_c)



def test():
    data_path='/root/TC174611125/fmaml/fmaml_mac/data/cifar10/center_data.npz'
    data = np.load(data_path)
    print(data['X'])

if __name__=="__main__":
    main()
    # test()