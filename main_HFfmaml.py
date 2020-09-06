import numpy as np
import argparse
import importlib
import random
import os
import tensorflow as tf
from flearn.utils.model_utils import read_data,load_weights,save_weights
from utils.utils import savemat

from flearn.models.client_HFmaml import Client
from tqdm import  tqdm

from scipy import io
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.reset_default_graph()

# GLOBAL PARAMETERS
OPTIMIZERS = ['HFfmaml','fmaml', 'fedavg', 'fedprox', 'feddane', 'fedddane', 'fedsgd']
DATASETS = ['sent140', 'nist', 'shakespeare', 'mnist',
'synthetic_iid', 'synthetic_0_0', 'synthetic_0.5_0.5', 'synthetic_1_1','cifar10','cifar100','Fmnist']  # NIST is EMNIST in the paepr


MODEL_PARAMS = {
    'sent140.bag_dnn': (2,), # num_classes
    'sent140.stacked_lstm': (25, 2, 100), # seq_len, num_classes, num_hidden 
    'sent140.stacked_lstm_no_embeddings': (25, 2, 100), # seq_len, num_classes, num_hidden
    'nist.mclr': (62,),  # num_classes, should be changed to 62 when using EMNIST
    'mnist.mclr': (10), # num_classes change
    'mnist.mclr2': (10),
    'Fmnist.mclr2': (10),
    'Fmnist.cnn': (10),
    'mnist.cnn': (10,),  # num_classes
    'cifar10.cnn': (10,),
    'cifar100.cnn': (100,),
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
    parser.add_argument('--num_rounds',default=10,help='number of rounds to simulate;',type=int)
    parser.add_argument('--eval_every',default=1,help='evaluate every rounds;',type=int)
    parser.add_argument('--clients_per_round',default=40,help='number of clients trained per round;',type=int)
    parser.add_argument('--batch_size',default=100,help='batch size when clients train on data;',type=int)
    parser.add_argument('--num_epochs',default=1,help='number of epochs when clients train on data;',type=int) #20
    parser.add_argument('--alpha',default=0.01,help='learning rate for inner solver;',type=float)
    parser.add_argument('--beta',default=0.003,help='meta rate for inner solver;',type=float)
    # parser.add_argument('--mu',help='constant for prox;',type=float,default=0.01)
    parser.add_argument('--seed',default=0,help='seed for randomness;',type=int)
    parser.add_argument('--labmda',default=1,help='labmda for regularizer',type=float)
    parser.add_argument('--rho',default=0.35,help='rho for regularizer',type=float)
    parser.add_argument('--mu_i',default=0,help='mu_i for optimizer',type=int)
    parser.add_argument('--adapt_num', default=1, help='adapt number', type=int)
    parser.add_argument('--isTrain', default=False, help='load trained wights', type=bool)
    parser.add_argument('--pretrain', default=False, help='Pretrain to get theta_c', type=bool)
    parser.add_argument('--sourceN', default=False, help='source node class num used', type=int)
    parser.add_argument('--R', default=0, help='the R th test', type=int)


    try: parsed = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))

    # Set seeds------
    # random.seed(1 + parsed['seed'])
    # np.random.seed(12 + parsed['seed'])
    # tf.set_random_seed(123 + parsed['seed'])


    # load selected model
    if parsed['dataset'].startswith("synthetic"):  # all synthetic_fed datasets use the same model
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', 'synthetic', parsed['model']) #changed
    elif  parsed['dataset']=="cifar10":
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', 'cifar10', parsed['model'])  # changed
    elif  parsed['dataset']=="cifar100":
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', 'cifar100', parsed['model'])  # changed
    elif  parsed['dataset']=="Fmnist":
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', 'Fmnist', parsed['model'])  # changed
    else:
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', 'mnist', parsed['model']) #parsed['dataset']

    print('@line 81 model path:',model_path)
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

def reshape_label(label,n=10):
    #print(label)
    new_label=[0]*n
    new_label[int(label)]=1
    return new_label

def reshape_features(x):
    x=np.array(x)
    x=np.transpose(x.reshape(3, 32, 32), [1, 2, 0])
    # print(x.shape)
    return x

def reshapeFmnist(x):
    x=np.array(x)
    x=x.reshape(28,28,1)
    return x

def prepare_dataset(options):
    # read data
    if options['dataset']=='cifar10' or options['dataset']=='cifar100':
        # data_path = os.path.join('data', options['dataset'], 'data')
        # dataset = read_data_xin(data_path)  # return clients, groups, train_data, test_data
        train_path = os.path.join('data', options['dataset'], 'data', 'train')
        test_path = os.path.join('data', options['dataset'], 'data', 'test')
        if options['pretrain']:
            train_path = os.path.join('data', options['dataset'], 'data', 'pretrain')
            test_path = os.path.join('data', options['dataset'], 'data', 'pretest')
        dataset = read_data(train_path, test_path)
        num_class = 10
        if options['dataset'] == 'cifar100':
            num_class = 100

        for user in dataset[0]:
            for i in range(len(dataset[2][user]['y'])):
                dataset[2][user]['x'][i]=reshape_features(dataset[2][user]['x'][i])
                dataset[2][user]['y'][i] = reshape_label(dataset[2][user]['y'][i],num_class)

        # print('reshape labels in test dataset')
        for user in dataset[0]:
            for i in range(len(dataset[3][user]['y'])):
                dataset[3][user]['x'][i] = reshape_features(dataset[3][user]['x'][i])
                dataset[3][user]['y'][i] = reshape_label(dataset[3][user]['y'][i],num_class)
    elif options['dataset']=='Fmnist':
        train_path = os.path.join('data', options['dataset'], 'data', 'train')
        test_path = os.path.join('data', options['dataset'], 'data', 'test')
        dataset = read_data(train_path, test_path) # return clients, groups, train_data, test_data
        for user in dataset[0]:
            for i in range(len(dataset[2][user]['y'])):
                dataset[2][user]['x'][i] = reshapeFmnist(dataset[2][user]['x'][i])
                dataset[2][user]['y'][i] = reshape_label(dataset[2][user]['y'][i])
            for i in range(len(dataset[3][user]['y'])):
                dataset[3][user]['x'][i] = reshapeFmnist(dataset[3][user]['x'][i])
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
            for i in range(len(dataset[3][user]['y'])):
                dataset[3][user]['y'][i] = reshape_label(dataset[3][user]['y'][i])
    random.seed(1)
    random.shuffle(dataset[0])
    test_user=dataset[0][options['clients_per_round']:]
    print('@ main print test user:',test_user)

    del dataset[0][options['clients_per_round']:]

    return test_user, dataset

def main():

    # suppress tf warnings
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    # parse command line arguments
    options, learner, optimizer = read_options()

    test_user,dataset=prepare_dataset(options)

    # define theta_c save path
    # theta_c_path='/root/TC174611125/fmaml/fmaml_mac/theta_c/{}_theata_c.mat'.format(options['dataset'])
    theta_c_path = '/root/TC174611125/fmaml/fmaml_mac/theta_c/{}_theata_c.mat'.format(options['dataset'])
    dir_path = os.path.dirname(theta_c_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    #、 o00000007理论 call appropriate trainer
    loss_save_path='./log/losses_OPT_{}_Dataset{}_round_{}_rho_{}_lambda{}_pretrain{}_SN{}.mat'.format(options['optimizer'],options['dataset'],options['num_rounds'],options['rho'],options['labmda'],options['pretrain'],options['sourceN'])
    acc_save_path='./log/Accuracies_OPT_{}_Dataset{}_round_{}_rho_{}_lambda{}_pretrain{}_SN{}.mat'.format(options['optimizer'],options['dataset'],options['num_rounds'],options['rho'],options['labmda'],options['pretrain'],options['sourceN'])
    if options['isTrain']==True:
        t = optimizer(options, learner, dataset,theta_c_path,test_user)
        loss_history,acc_history=t.train()
        savemat(loss_save_path, {'losses': loss_history})
        savemat(acc_save_path,{'accuracies':acc_history})
        print('Finished training')
        client_params = t.latest_model
        weight = client_params
    else:
        # weight=load_weights('{}_{}_trained_weights.mat'.format(options['dataset'],options['model']))
        weight = load_weights(theta_c_path)

    #save theta_c
    if options['pretrain'] and options['isTrain']:
        print('######################### Saving pretrained thetaC .............>>>>>>>>>>>>>>>>>')
        w_names = t.client_model.get_param_names()
        save_weights(weight, w_names,theta_c_path)

    loss_test, acc_test = target_test(test_user,learner,dataset,options,weight)
    tqdm.write(' Final loss: {}'.format(np.sum(loss_test)))
    print("Local average acc", np.sum(acc_test))
    print("loss_save_path:", loss_save_path)
    print("acc_save_path:", acc_save_path)
    # save_result('./results/ThetaC_results.csv',[[options['labmda'],np.sum(acc_test),acc_save_path]],col_name=['Lambda','Accuracy','acc_save_path'])
    save_result('./results/contrast_{}_{}.csv'.format(options['dataset'],['optimizer']), [[options['labmda'], np.sum(acc_test), acc_save_path]],
                col_name=['Lambda', 'Accuracy', 'acc_save_path'])


def target_test(test_user,learner,dataset,options,weight):
    loss_test=dict()
    accs=dict()
    num_test=dict()
    for i,user in enumerate(test_user):
        loss_test[i],accs[i],num_test[i]=fmaml_test(learner=learner, train_data=dataset[2][user], test_data=dataset[3][user],
                params=options, user_name=user, weight= weight)
    loss_test=list(loss_test.values())
    accs=list(accs.values())
    num_test=list(num_test.values())
    loss_test = [l*n/np.sum(num_test) for l,n in zip(loss_test,num_test)]
    acc_test = [a * n/np.sum(num_test) for a, n in zip(accs, num_test)]
    return loss_test,acc_test

def fmaml_test(learner, train_data, test_data, params, user_name, weight):
    print('HFmaml test')
    params['w_i']=1
    #client_params = trainer.latest_model
    client_model = learner(params)  # changed remove star

    test_client = Client(user_name, [], train_data, test_data, client_model)
    test_client.set_params(weight)

    _ = test_client.fast_adapt(params['adapt_num'])

    acc, test_loss, test_num,preds = test_client.test_test()

    # print('@main_HFmaml line 240 preds:',preds)

    return test_loss,acc,test_num


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

if __name__ == '__main__':
    main()
