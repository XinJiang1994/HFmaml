import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf
import os

from .fedbase_maml import BaseFedarated
#from flearn.models.client_maml import Client
#from flearn.utils.model_utils import Metrics
from flearn.utils.tf_utils import process_grad


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated MAML to Train')
        self.opt1 = params['learning_rate']
        self.opt2 = params['meta_rate']
        self.labmda=params['labmda']
        _, _, self.train_data, self.test_data = dataset
        #self.inner_opt = zip(self.inner_opt1, self.inner_opt2)
        super(Server, self).__init__(params, learner, dataset)
        ### @xinjiang
        # set theta_c
        ### end
        self.set_theta_c(params)



    def train(self):
        '''Train using Federated MAML'''
        print('Training with {} workers ---'.format(self.clients_per_round))
        for i in trange(self.num_rounds, desc='Round: ', ncols=120):
            # test model
            if i % self.eval_every == 0:
                stats = self.test()
                stats_train = self.train_error_and_loss()
                self.metrics.accuracies.append(stats)
                self.metrics.train_accuracies.append(stats_train)
                tqdm.write('At round {} accuracy: {}'.format(i, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))
                tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(stats_train[3]) * 1.0 / np.sum(
                    stats_train[2])))
                tqdm.write('At round {} training loss: {}'.format(i,
                                                                  np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(
                                                                      stats_train[2])))

                model_len = process_grad(self.latest_model).size
                global_grads = np.zeros(model_len)
                client_grads = np.zeros(model_len)
                num_samples = []
                local_grads = []

                for c in self.clients:
                    num, client_grad = c.get_grads(model_len)
                    local_grads.append(client_grad)
                    num_samples.append(num)
                    global_grads = np.add(global_grads, client_grads * num)
                global_grads = global_grads * 1.0 / np.sum(np.asarray(num_samples))

                difference = 0
                for idx in range(len(self.clients)):
                    #difference += np.sum(np.square(global_grads - local_grads[idx]))
                    difference = 0
                difference = difference * 1.0 / len(self.clients)
                tqdm.write('gradient difference: {}'.format(difference))

               #  save server model
                self.metrics.write()
                self.save()

            # choose M clients prop to data size, here need to choose all
            selected_clients = self.select_clients(i, num_clients=self.clients_per_round)

            csolns = [] # buffer for receiving client solutions
            for c in tqdm(selected_clients, desc='Client: ', leave=False, ncols=120):
                # communicate the latest model
                c.set_params(self.latest_model)

                # solve minimization locally
                soln, stats = c.solve_inner(num_epochs=self.num_epochs)
                # gather solutions from client
                csolns.append(soln)

                # track communication cost
                self.metrics.update(rnd=i, cid=c.id, stats=stats)

                # update model
            self.latest_model = self.aggregate(csolns)

            # final test model
        stats = self.test()
        stats_train = self.train_error_and_loss()

        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)
        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds,
                                                                  np.sum(stats_train[3]) * 1.0 / np.sum(
                                                                      stats_train[2])))
        # save server model
        self.metrics.write()
        self.save()
    ##@xinjiang
    def set_theta_c(self,params):
        theta_c = []
        for it, par in enumerate(self.client_model.get_params()):
            seed = 133 + params['seed'] + it
            ## the shape of params might different, for example: w--(784，10), b--(10,)
            len_c = 1
            for s in par.shape:
                len_c *= s
            np.random.seed(seed)
            th_c_flat = np.random.rand(len_c)
            th_c = th_c_flat.reshape(par.shape)
            # print('th_c.shape',th_c.shape)
            th_c=np.zeros_like(par)
            theta_c.append(th_c)
        self.theta_c=theta_c
        # s0,s1=self.client_model.get_params()[0].shape
        # print('xinajiang?????????', params['seed'])
        # os.system("pause");
