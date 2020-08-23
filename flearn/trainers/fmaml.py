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
        _, _, self.train_data, self.test_data = dataset
        #self.inner_opt = zip(self.inner_opt1, self.inner_opt2)
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        '''Train using Federated MAML'''
        print('Training with {} workers ---'.format(self.clients_per_round))
        print('@fmaml line28 num_rounds',self.num_rounds)
        print('@fmaml line29 self.clients_per_round',self.clients_per_round)
        for i in trange(self.num_rounds, desc='Round: ', ncols=120):
            # test model
            if i % self.eval_every == 0:
                stats = self.test()
                stats_train = self.train_error_and_loss()
                self.metrics.accuracies.append(stats)
                self.metrics.train_accuracies.append(stats_train)
                # tqdm.write('At round {} accuracy: {}'.format(i, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))
                # tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(stats_train[3]) * 1.0 / np.sum(
                #     stats_train[2])))
                tqdm.write('At round {} training loss: {}'.format(i,
                                                                  np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(
                                                                      stats_train[2])))
               #  save server model
                self.metrics.write()
                self.save()

            # choose M clients prop to data size, here need to choose all
            selected_clients = self.select_clients(i, num_clients=self.clients_per_round)

            csolns = [] # buffer for receiving client solutions
            for c in selected_clients:
                # communicate the latest model
                c.set_params(self.latest_model)

                # solve minimization locally
                soln= c.solve_inner(num_epochs=self.num_epochs)
                # gather solutions from client
                csolns.append(soln)

                # track communication cost
                # self.metrics.update(rnd=i, cid=c.id, stats=stats)

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
        tqdm.write('At round {} training loss: {}'.format(self.num_rounds, np.mean(stats_train[4])))
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
