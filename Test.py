from scipy import io
import numpy as np

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

import tensorflow as tf

x=tf.convert_to_tensor(np.array([[-1.0,-10,-2.5,0.0,-1.0,-5.1,-5.1,-45,-1.2,-3.5], [1.0,-10,2.5,0.0,1.0,5.1,1.3,-145,1.2,3.5]]))
y=tf.nn.softmax(x)
pred=tf.argmax(y,axis=1)
pred2=tf.argmax(x,axis=1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(y))
    print(sess.run(pred))
    print(sess.run(pred2))