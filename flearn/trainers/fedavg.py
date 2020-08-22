import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated Average to Train')
        #self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        self.opt1 = params['alpha']
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        '''Train using Federated Averaging'''
        print('Training with {} workers ---'.format(self.clients_per_round))
        for i in trange(self.num_rounds, desc='Round: ', ncols=120):
            # test model
            if i % self.eval_every == 0:
                stats = self.test()
                stats_train = self.train_error_and_loss()
                self.metrics.accuracies.append(stats)
                self.metrics.train_accuracies.append(stats_train)
                # tqdm.write('At round {} accuracy: {}'.format(i, np.sum(stats[3])*1.0/np.sum(stats[2])))
                # tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))
                tqdm.write('At round {} training loss: {}'.format(i, np.dot(stats_train[4], stats_train[2])*1.0/np.sum(stats_train[2])))

                # save server model
                self.metrics.write()
                self.save()

            # choose K clients prop to data size
            selected_clients = self.select_clients(i, num_clients=self.clients_per_round)
            print('Training with {} workers ---'.format(len(selected_clients)))

            csolns = [] # buffer for receiving client solutions
            for c in selected_clients:
                # communicate the latest model
                c.set_params(self.latest_model)

                # solve minimization locally
                soln = c.solve_inner(num_epochs=self.num_epochs)#, batch_size=self.batch_size)

                # gather solutions from client
                csolns.append(soln)

                # track communication cost
                #self.metrics.update(rnd=i, cid=c.id, stats=stats)
        
            # update model
            self.latest_model = self.aggregate(csolns)

        # final test model
        stats = self.test()
        stats_train = self.train_error_and_loss()

        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)
        # tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3])*1.0/np.sum(stats[2])))
        # tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))
        # save server model
        self.metrics.write()
        self.save()
