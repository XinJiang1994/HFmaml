import numpy as np
import argparse
import importlib
import random
import os
import tensorflow as tf
from flearn.utils.model_utils import read_data

from flearn.models.client_maml import Client
from tqdm import trange, tqdm

# GLOBAL PARAMETERS
OPTIMIZERS = ['fmaml', 'fedavg', 'fedprox', 'feddane', 'fedddane', 'fedsgd']
DATASETS = ['sent140', 'nist', 'shakespeare', 'mnist',
'synthetic_iid', 'synthetic_0_0', 'synthetic_0.5_0.5', 'synthetic_1_1']  # NIST is EMNIST in the paepr


MODEL_PARAMS = {
    'sent140.bag_dnn': (2,), # num_classes
    'sent140.stacked_lstm': (25, 2, 100), # seq_len, num_classes, num_hidden 
    'sent140.stacked_lstm_no_embeddings': (25, 2, 100), # seq_len, num_classes, num_hidden
    'nist.mclr': (62,),  # num_classes, should be changed to 62 when using EMNIST
    'mnist.mclr': (10), # num_classes change
    'mnist.cnn': (10,),  # num_classes
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
                    default='mnist')
    parser.add_argument('--model',
                    help='name of model;',
                    type=str,
                    default='mclr')
    parser.add_argument('--num_rounds',
                    help='number of rounds to simulate;',
                    type=int,
                    default=150)
    parser.add_argument('--eval_every',
                    help='evaluate every ____ rounds;',
                    type=int,
                    default=1)
    parser.add_argument('--clients_per_round',
                    help='number of clients trained per round;',
                    type=int,
                    default=50)
    parser.add_argument('--batch_size',
                    help='batch size when clients train on data;',
                    type=int,
                    default=10)
    parser.add_argument('--num_epochs', 
                    help='number of epochs when clients train on data;',
                    type=int,
                    default=1) #20
    parser.add_argument('--alpha',
                    help='learning rate for inner solver;',
                    type=float,
                    default=0.01)
    parser.add_argument('--beta',
                    help='meta rate for inner solver;',
                    type=float,
                    default=0.003)
    parser.add_argument('--mu',
                    help='constant for prox;',
                    type=float,
                    default=0.01)
    parser.add_argument('--seed',
                    help='seed for randomness;',
                    type=int,
                    default=0)
    parser.add_argument('--num_local_updates',
                    help='number of rounds to simulate;',
                    type=int,
                    default=1)

    try: parsed = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))

    # Set seeds
    random.seed(1 + parsed['seed'])
    np.random.seed(12 + parsed['seed'])
    tf.set_random_seed(123 + parsed['seed'])


    # load selected model
    if parsed['dataset'].startswith("synthetic"):  # all synthetic_fed datasets use the same model
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', 'synthetic', parsed['model']) #changed
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
    if label == 0:
        new_label = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if label == 1:
        new_label = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if label == 2:
        new_label = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if label == 3:
        new_label = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if label == 4:
        new_label = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if label == 5:
        new_label = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    if label == 6:
        new_label = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    if label == 7:
        new_label = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    if label == 8:
        new_label = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    if label == 9:
        new_label = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    return new_label

def main():
    # suppress tf warnings
    tf.logging.set_verbosity(tf.logging.WARN)
    
    # parse command line arguments
    options, learner, optimizer = read_options()

    # read data
    train_path = os.path.join('data', options['dataset'], 'data', 'train')
    test_path = os.path.join('data', options['dataset'], 'data', 'test')
    dataset = read_data(train_path, test_path)

    for user in dataset[0]:
        for i in range(len(dataset[2][user]['y'])):
            dataset[2][user]['y'][i] = reshape_label(dataset[2][user]['y'][i])

    #print('reshape labels in test dataset')
    for user in dataset[0]:
        for i in range(len(dataset[3][user]['y'])):
            dataset[3][user]['y'][i] = reshape_label(dataset[3][user]['y'][i])



    num_users = len(dataset[0])
    # print('num users: ',num_users)
    test_user = dataset[0][79:num_users-1]
    del dataset[0][79:num_users-1]


    # call appropriate trainer
    t = optimizer(options, learner, dataset)
    t.train()

    print('after training, start testing')

    client_params = t.latest_model
    weight = client_params
    # client_model = learner(options['model_params'], options['learning_rate'], options['meta_rate'],
    #                        dataset[2][test_user], dataset[3][test_user])  # changed remove star
    # client_model.set_params(client_params)
    # test_client = Client(test_user, [], dataset[2][test_user], dataset[3][test_user], client_model)
    # loss_zero = test_client.test_zeroth()
    # tqdm.write('Zeroth loss {}:'.format(loss_zero))

    local_updates_bound = 2
    for num_local_updates in range(1, local_updates_bound):
        i=0
        loss_test=dict()
        accs = dict()
        for user in test_user:
            accs[i],loss_test[i]=fmaml_test(trainer=t, learner=learner, train_data=dataset[2][user], test_data=dataset[3][user],
                   params=options, user_name=user, num_local_updates=num_local_updates, weight= weight)
            i=i+1
        loss_test = sum(loss_test.values()) / len(loss_test)
        accs = sum(accs.values()) / len(accs)
        tqdm.write('Local updates {} Final loss: {}'.format(num_local_updates, loss_test))
        tqdm.write('Local updates {} Final acc: {}'.format(num_local_updates, accs))


def fmaml_test(trainer, learner, train_data, test_data, params, user_name, num_local_updates, weight):
    print('fmaml test')

    #client_params = trainer.latest_model
    client_model = learner(params)  # changed remove star
    #client_model.set_params(weight)

    #test_client = Client(user_name, [], train_data, test_data, client_model)
    #trainer.client_model.set_params(weight)
    # client_params = trainer.latest_model

    test_client = Client(user_name, [], train_data, test_data, client_model)
    test_client.set_params(weight)

    # tot_correct, loss, test_loss, ns = test_client.final_test()
    # client_params = trainer.latest_model
    soln = test_client.fast_adapt(1)

    test_client.set_params(soln)

    acc, test_loss, _ = test_client.test_test()

    return acc,test_loss
    
if __name__ == '__main__':
    main()
