import numpy as np
import argparse
import importlib
import random
import os
import tensorflow as tf
from flearn.utils.model_utils import read_data,read_data_xin

from flearn.models.client_HFmaml import Client
from tqdm import  tqdm

# GLOBAL PARAMETERS
OPTIMIZERS = ['HFfmaml','fmaml', 'fedavg', 'fedprox', 'feddane', 'fedddane', 'fedsgd']
DATASETS = ['sent140', 'nist', 'shakespeare', 'mnist',
'synthetic_iid', 'synthetic_0_0', 'synthetic_0.5_0.5', 'synthetic_1_1']  # NIST is EMNIST in the paepr


MODEL_PARAMS = {
    'sent140.bag_dnn': (2,), # num_classes
    'sent140.stacked_lstm': (25, 2, 100), # seq_len, num_classes, num_hidden 
    'sent140.stacked_lstm_no_embeddings': (25, 2, 100), # seq_len, num_classes, num_hidden
    'nist.mclr': (62,),  # num_classes, should be changed to 62 when using EMNIST
    'mnist.mclr': (10), # num_classes change
    'mnist.mclr2': (10),
    'mnist.cnn': (10,),  # num_classes
    'cifar10.cnn': (10,),
    'shakespeare.stacked_lstm': (80, 80, 256), # seq_len, emb_dim, num_hidden
    'synthetic_fed.mclr': (10), # num_classes changed, remove,
    'synthetic.mclr2': (10), # num_classes changed, remove,
}


def read_options():
    ''' Parse command line arguments or load defaults '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer',default='HFfmaml',help='name of optimizer;',type=str,choices=OPTIMIZERS)
    parser.add_argument('--dataset',default='cifar10',help='name of dataset;',type=str,choices=DATASETS)
    parser.add_argument('--model',default='cnn',help='name of model;',type=str)
    parser.add_argument('--num_rounds',default=800,help='number of rounds to simulate;',type=int)
    parser.add_argument('--eval_every',default=1,help='evaluate every rounds;',type=int)
    parser.add_argument('--clients_per_round',default=80,help='number of clients trained per round;',type=int)
    parser.add_argument('--batch_size',default=10,help='batch size when clients train on data;',type=int)
    parser.add_argument('--num_epochs',default=5,help='number of epochs when clients train on data;',type=int) #20
    parser.add_argument('--alpha',default=0.01,help='learning rate for inner solver;',type=float)
    parser.add_argument('--beta',default=0.000000000000001,help='meta rate for inner solver;',type=float)
    # parser.add_argument('--mu',help='constant for prox;',type=float,default=0.01)
    parser.add_argument('--seed',default=0,help='seed for randomness;',type=int)
    parser.add_argument('--labmda',default=0,help='labmda for regularizer',type=int)
    parser.add_argument('--rho',default=25,help='rho for regularizer',type=int)
    parser.add_argument('--mu_i',default=0,help='mu_i for optimizer',type=int)

    try: parsed = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))

    # Set seeds------
    random.seed(1 + parsed['seed'])
    np.random.seed(12 + parsed['seed'])
    tf.set_random_seed(123 + parsed['seed'])


    # load selected model
    if parsed['dataset'].startswith("synthetic"):  # all synthetic_fed datasets use the same model
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', 'synthetic', parsed['model']) #changed
    elif  parsed['dataset'].startswith("cifar10"):
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', 'cifar10', parsed['model'])  # changed
    else:
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', 'mnist', parsed['model']) #parsed['dataset']

    mod = importlib.import_module(model_path)
    learner = getattr(mod, 'Model')

    # load selected trainer
    opt_path = 'flearn.trainers.%s' % parsed['optimizer']
    mod = importlib.import_module(opt_path)
    optimizer = getattr(mod, 'Server')

    # add selected model parameter
    parsed['num_classes'] = MODEL_PARAMS['.'.join(model_path.split('.')[2:])]

    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()]);
    fmtString = '\t%' + str(maxLen) + 's : %s';
    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)

    return parsed, learner, optimizer

def reshape_label(label):
    #print(label)
    new_label=[0]*10
    new_label[int(label)]=1
    return new_label

def reshape_features(x):
    x=np.array(x)
    x=np.transpose(x.reshape(3, 32, 32), [1, 2, 0])
    # print(x.shape)
    return x

def main():
    # suppress tf warnings
    tf.logging.set_verbosity(tf.logging.WARN)
    
    # parse command line arguments
    options, learner, optimizer = read_options()

    # read data
    if options['dataset']=='cifar10':
        data_path = os.path.join('data', options['dataset'], 'data')
        # dataset = read_data_xin(data_path)  # return clients, groups, train_data, test_data
        train_path = os.path.join('data', options['dataset'], 'data', 'train')
        test_path = os.path.join('data', options['dataset'], 'data', 'test')
        dataset = read_data(train_path, test_path)

        for user in dataset[0]:
            for i in range(len(dataset[2][user]['y'])):
                dataset[2][user]['x'][i]=reshape_features(dataset[2][user]['x'][i])
                dataset[2][user]['y'][i] = reshape_label(dataset[2][user]['y'][i])

        # print('reshape labels in test dataset')
        for user in dataset[0]:
            for i in range(len(dataset[3][user]['y'])):
                dataset[3][user]['x'][i] = reshape_features(dataset[3][user]['x'][i])
                dataset[3][user]['y'][i] = reshape_label(dataset[3][user]['y'][i])
    else:
        train_path = os.path.join('data', options['dataset'], 'data', 'train')
        test_path = os.path.join('data', options['dataset'], 'data', 'test')
        dataset = read_data(train_path, test_path) # return clients, groups, train_data, test_data
        #print(dataset[3]['f_00000']['y'])
        #print('@main_HFfaml.py line 152####',dataset)

        for user in dataset[0]:
            for i in range(len(dataset[2][user]['y'])):
                dataset[2][user]['y'][i] = reshape_label(dataset[2][user]['y'][i])

        #print('reshape labels in test dataset')
        for user in dataset[0]:
            for i in range(len(dataset[3][user]['y'])):
                dataset[3][user]['y'][i] = reshape_label(dataset[3][user]['y'][i])
    random.shuffle(dataset[0])
    test_user=dataset[0][180:]
    del dataset[0][180:]

    sams_train=[]
    sams_taget=[]
    for user in dataset[0]:
        sams_train.extend(dataset[2][user]['y'])
        sams_train.extend(dataset[3][user]['y'])
    for user in test_user:
        sams_taget.extend(dataset[2][user]['y'])
        sams_taget.extend(dataset[3][user]['y'])

    sams_train=[np.argmax(x) for x in sams_train]
    sams_taget = [np.argmax(x) for x in sams_taget]
    #print(sams_train)

    import collections
    c_train=collections.Counter(sams_train)
    c_target=collections.Counter(sams_taget)

    print(c_target)

    p1={}
    p2={}
    s1=0
    s2=0
    for i in range(10):
        s1 += c_train[i]
        s2 += c_target[i]
    for i in range(10):
        #print(i)
        p1[i]=c_train[i]/s1
        p2[i]=c_target[i]/s2

    print(p1)
    print(p2)

    #、 o00000007理论 call appropriate trainer
    t = optimizer(options, learner, dataset)
    loss_history=t.train()
    plot_losses(loss_history)


    print('after training, start testing')

    client_params = t.latest_model
    weight = client_params

    loss_test=dict()
    accs=dict()
    preds=dict()
    for i,user in enumerate(test_user):
        #print(dataset[2][user])
        loss_test[i],accs[i],preds[i]=fmaml_test(trainer=t, learner=learner, train_data=dataset[2][user], test_data=dataset[3][user],
                params=options, user_name=user, weight= weight)
    loss_test = np.mean(list(loss_test.values()))
    tqdm.write(' Final loss: {}'.format(loss_test))
    print("Local average acc",np.mean(list(accs.values())))
    # for i,user in enumerate(test_user):
    #     print(user)
    #     test_data=dataset[3][user]
    #     ys=[]
    #     for y in (test_data['y']):
    #         ys.append(np.argmax(y))
    #     print(ys)
    #     print(len(ys))
    #print(preds)

def fmaml_test(trainer, learner, train_data, test_data, params, user_name, weight):
    print('fmaml test')

    #client_params = trainer.latest_model
    client_model = learner(params)  # changed remove star


    test_client = Client(user_name, [], train_data, test_data, client_model)
    test_client.set_params(weight)
    #r = np.load('/root/TC174611125/fmaml/HFmaml/weights.npz')

    #print('################:',np.sum(weight[0]-r['w']))

    # tot_correct, loss, test_loss, ns = test_client.final_test()
    soln = test_client.fast_adapt(1)

    #np.savez('weights.npz',w=soln[0],b=soln[1])
    # soln = weight
    test_client.set_params(soln)

    acc, test_loss, samp_num,preds = test_client.test_test()

    return test_loss,acc,preds

import matplotlib.pyplot as plt
def plot_losses(losses):
    plt.plot(losses)
    
if __name__ == '__main__':
    main()
