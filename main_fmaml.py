import numpy as np
import argparse
import importlib
import random
import os
import tensorflow as tf
from flearn.utils.model_utils import read_data

from flearn.models.client_maml import Client
from tqdm import trange, tqdm
from scipy import io

from main_HFfmaml import prepare_dataset, target_test
from utils.utils import savemat, save_result

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# GLOBAL PARAMETERS
OPTIMIZERS = ['fmaml', 'fedavg', 'fedprox', 'feddane', 'fedddane', 'fedsgd']
DATASETS = ['sent140', 'nist', 'shakespeare', 'mnist',
'synthetic_iid', 'synthetic_0_0', 'synthetic_0.5_0.5', 'synthetic_1_1','cifar10','cifar100','Fmnist']  # NIST is EMNIST in the paepr


MODEL_PARAMS = {
    'sent140.bag_dnn': (2,), # num_classes
    'sent140.stacked_lstm': (25, 2, 100), # seq_len, num_classes, num_hidden 
    'sent140.stacked_lstm_no_embeddings': (25, 2, 100), # seq_len, num_classes, num_hidden
    'nist.mclr': (62,),  # num_classes, should be changed to 62 when using EMNIST
    'mnist.mclr': (10), # num_classes change
    'Fmnist.mclr': (10),
    'Fmnist.cnn_fmaml': (10,),
    'mnist.cnn': (10,),  # num_classes
    'cifar100.cnn_fmaml': (100,),
    'cifar10.cnn_fmaml': (10,),
    'shakespeare.stacked_lstm': (80, 80, 256), # seq_len, emb_dim, num_hidden
    'synthetic_fed.mclr': (10), # num_classes changed, remove,
    'synthetic.mclr': (10), # num_classes changed, remove,
}


def read_options():
    ''' Parse command line arguments or load defaults '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer',
                    help='name of optimizer;',
                    type=str,
                    choices=OPTIMIZERS,
                    default='fmaml')
    parser.add_argument('--dataset',
                    help='name of dataset;',
                    type=str,
                    choices=DATASETS,
                    default='cifar100')
    parser.add_argument('--model',
                    help='name of model;',
                    type=str,
                    default='cnn_fmaml')
    parser.add_argument('--num_rounds',
                    help='number of rounds to simulate;',
                    type=int,
                    default=50)
    parser.add_argument('--eval_every',
                    help='evaluate every ____ rounds;',
                    type=int,
                    default=1)
    parser.add_argument('--clients_per_round',
                    help='number of clients trained per round;',
                    type=int,
                    default=40)
    parser.add_argument('--batch_size',
                    help='batch size when clients train on data;',
                    type=int,
                    default=10)
    parser.add_argument('--num_epochs', 
                    help='number of epochs when clients train on data;',
                    type=int,
                    default=10) #20
    parser.add_argument('--alpha',
                    help='learning rate for inner solver;',
                    type=float,
                    default=0.01)
    parser.add_argument('--beta',
                    help='meta rate for inner solver;',
                    type=float,
                    default=0.01)
    parser.add_argument('--mu',
                    help='constant for prox;',
                    type=float,
                    default=0.01)
    parser.add_argument('--seed',
                    help='seed for randomness;',
                    type=int,
                    default=0)
    parser.add_argument('--adapt_num', default=1, help='mu_i for optimizer', type=int)

    parser.add_argument('--R', default=0, help='the R th test', type=int)
    parser.add_argument('--logdir', default='./log', help='the R th test', type=str)
    parser.add_argument('--pretrain', default=False, help='Pretrain to get theta_c', type=bool)
    parser.add_argument('--transfer', default=False, help='Pretrain to get theta_c', type=bool)


    try: parsed = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))

    # Set seeds
    random.seed(1 + parsed['seed'])
    np.random.seed(12 + parsed['seed'])
    tf.set_random_seed(123 + parsed['seed'])


    # load selected model
    if parsed['dataset'].startswith("synthetic"):  # all synthetic_fed datasets use the same model
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', 'synthetic', parsed['model']) #changed
    elif parsed['dataset']=="cifar10":
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', 'cifar10', parsed['model'])
    elif parsed['dataset']=="cifar100":
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', 'cifar100', parsed['model'])
    elif parsed['dataset'].startswith("Fmnist"):
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', 'Fmnist', parsed['model'])
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

def reshape_label(label,n=10):
    # print(label)
    new_label = [0] * n
    new_label[int(label)] = 1
    return new_label


def reshape_features(x):
    x = np.array(x)
    x = np.transpose(x.reshape(3, 32, 32), [1, 2, 0])
    # print(x.shape)
    return x

def reshapeFmnist(x):
    x=np.array(x)
    x=x.reshape(28,28,1)
    return x


def main():
    tf.reset_default_graph()
    # suppress tf warnings
    tf.logging.set_verbosity(tf.logging.WARN)

    # parse command line arguments
    options, learner, optimizer = read_options()
    print('$$$$$$$$$$$$$$$$$learner',learner)

    test_user, dataset = prepare_dataset(options)
    # call appropriate trainer
    theta_c_path = '/root/TC174611125/fmaml/fmaml_mac/theta_c/{}_theata_c.mat'.format(options['dataset'])
    t = optimizer(options, learner,theta_c_path, dataset,test_user)
    loss_history,acc_history=t.train()
    loss_save_path='losses_OPT_{}_Dataset_{}_beta{}_round{}_L{}_R{}.mat'.format(options['optimizer'],
                                                                                options['dataset'],
                                                                                options['beta'],
                                                                                options['num_rounds'],
                                                                                options['num_epochs'],
                                                                                options['R'])
    acc_save_path = 'Accuracies_OPT_{}_Dataset_{}_beta{}_round{}_L{}_R{}.mat'.format(options['optimizer'],
                                                                                     options['dataset'],
                                                                                     options['beta'],
                                                                                     options['num_rounds'],
                                                                                     options['num_epochs'],
                                                                                     options['R'])
    loss_save_path=os.path.join(options['logdir'],loss_save_path)
    acc_save_path=os.path.join(options['logdir'],acc_save_path)

    savemat(loss_save_path, {'losses': loss_history})
    savemat(acc_save_path, {'accuracies': acc_history})

    print('after training, start testing')

    client_params = t.latest_model
    weight = client_params
    loss_test, acc_test = target_test(test_user,learner,dataset,options,weight)
    loss_test_forget, acc_test_forget = target_test(test_user, learner, dataset, options, weight,situation='forget_test')
    tqdm.write(' Final loss: {}'.format(np.sum(loss_test)))
    print("Local average acc", np.sum(acc_test))
    print("Forget_test average acc", np.sum(acc_test_forget))
    print("loss_save_path:", loss_save_path)
    print("acc_save_path:", acc_save_path)
    result_path=os.path.join(options['logdir'],'contrast_{}_{}_{}_L{}.csv'.format(options['model'],options['dataset'],options['optimizer'],options['num_epochs']))
    save_result(result_path, [[np.sum(acc_test),np.sum(acc_test_forget), acc_save_path,loss_save_path]],
                    col_name=['Accuracy','Forget_acc' ,'acc_save_path','loss_save_path'])
    
if __name__ == '__main__':
    main()
