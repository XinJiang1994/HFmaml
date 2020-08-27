from scipy import io
import numpy as np

def load_weights(wPath='weights.mat'):
    params=io.loadmat(wPath)
    vars=list(params.values())[3:]
    vars=[np.squeeze(x) for x in vars]
    #print('@HFmaml line 85',vars)
    return vars

params=io.loadmat('weights.mat')

print(params)

