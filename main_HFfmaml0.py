import numpy as np
import argparse
import importlib
import random
import os
import tensorflow as tf
from flearn.utils.model_utils import read_data

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
    'shakespeare.stacked_lstm': (80, 80, 256), # seq_len, emb_dim, num_hidden
    'synthetic_fed.mclr': (10), # num_classes changed, remove,
    'synthetic.mclr2': (10), # num_classes changed, remove,
}


def read_options():
    ''' Parse command line arguments or load defaults '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer',default='HFfmaml',help='name of optimizer;',type=str,choices=OPTIMIZERS)
    parser.add_argument('--dataset',default='synthetic_0_0',help='name of dataset;',type=str,choices=DATASETS)
    parser.add_argument('--model',default='mclr2',help='name of model;',type=str)
    parser.add_argument('--num_rounds',default=0,help='number of rounds to simulate;',type=int)
    parser.add_argument('--eval_every',default=1,help='evaluate every ____ rounds;',type=int)
    parser.add_argument('--clients_per_round',default=40,help='number of clients trained per round;',type=int)
    parser.add_argument('--batch_size',default=20,help='batch size when clients train on data;',type=int)
    parser.add_argument('--num_epochs',default=5,help='number of epochs when clients train on data;',type=int) #20
    parser.add_argument('--learning_rate',default=0.003,help='learning rate for inner solver;',type=float)
    parser.add_argument('--meta_rate',default=0.000000000000001,help='meta rate for inner solver;',type=float)
    # parser.add_argument('--mu',help='constant for prox;',type=float,default=0.01)
    parser.add_argument('--seed',default=0,help='seed for randomness;',type=int)
    parser.add_argument('--labmda',default=0,help='labmda for regularizer',type=int)
    parser.add_argument('--rho',default=0.6,help='rho for regularizer',type=int)
    parser.add_argument('--mu_i',default=0,help='mu_i for optimizer',type=int)

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
    parsed['model_params'] = MODEL_PARAMS['.'.join(model_path.split('.')[2:])]

    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()]);
    fmtString = '\t%' + str(maxLen) + 's : %s';
    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)

    return parsed, learner, optimizer

def reshape_label(label):
    new_label=[0]*10
    new_label[int(label)]=1
    return new_label

def main():
    # suppress tf warnings
    tf.logging.set_verbosity(tf.logging.WARN)
    
    # parse command line arguments
    options, learner, optimizer = read_options()

    # read data
    train_path = os.path.join('data', options['dataset'], 'data', 'train')
    test_path = os.path.join('data', options['dataset'], 'data', 'test')
    dataset = read_data(train_path, test_path) # return clients, groups, train_data, test_data

    #print('@main_HFfaml.py line 152####',dataset)

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
    client_model = learner(params['model_params'], params['learning_rate'])  # changed remove star


    test_client = Client(user_name, [], train_data, test_data, client_model)
    test_client.set_params(weight)

    # tot_correct, loss, test_loss, ns = test_client.final_test()
    soln = test_client.fast_adapt(1)
    #print('@main_HFfmaml line 233, soln:',soln)
    #np.savez('weights.npz', w=soln[0], b=soln[1])
    test_client.set_params(soln)

    tot_corect, test_loss, samp_num,preds = test_client.test_test()

    return test_loss,tot_corect/samp_num,preds
    
if __name__ == '__main__':
    main()
