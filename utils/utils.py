import pickle
import os
import pandas as pd
from scipy import io

def savemat(filepath,data):
    # 如果文件夹不存在则创建文件夹
    dir_name = os.path.dirname(filepath)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    io.savemat(filepath, data)


def save_result(filename,records,col_name = ['Lambda','Accuracy', 'AccSavePath']):
    # 如果文件夹不存在则创建文件夹
    dir_name=os.path.dirname(filename)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # 如果文件不存在，说明是第一次保存，先创建文件的title
    if not os.path.exists(filename):
        df_title=pd.DataFrame(data=[],columns=col_name)
        df_title.to_csv(filename, encoding='utf-8', mode='a', index=False)
    df = pd.DataFrame(data=records,columns=col_name)
    df.to_csv(filename, encoding='utf-8',mode='a', index=False,header=0)#不要保存header，不然会重复保存header
    print("The results have been successfully saved")

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def iid_divide(l, g):
    '''
    divide list l among g groups
    each group has either int(len(l)/g) or int(len(l)/g)+1 elements
    returns a list of groups
    '''
    num_elems = len(l)
    group_size = int(len(l)/g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l[group_size*i:group_size*(i+1)])
    bi = group_size*num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi+group_size*i:bi+group_size*(i+1)])
    return glist