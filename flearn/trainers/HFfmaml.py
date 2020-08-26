import numpy as np
from tqdm import trange, tqdm

from .fedbase_HFmaml import BaseFedarated

class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):

        print('Using Federated MAML to Train')
        self.lamda=params['labmda']
        _, _, self.train_data, self.test_data = dataset
        super(Server, self).__init__(params, learner, dataset)
        ### @xinjiang set theta_c ### end
        self.set_theta_c()

    def train(self):
        print('Training with {} workers ---'.format(self.clients_per_round))
        ## num_rounds is k
        ## num_epochs should set 1 in HFfmaml
        loss_history=[]
        for i in trange(self.num_rounds, desc='Round: ', ncols=120):
            # test model
            if i % self.eval_every == 0:
                stats = self.test()
                stats_train = self.train_error_and_loss()
                # print(stats_train)
                self.metrics.accuracies.append(stats)
                self.metrics.train_accuracies.append(stats_train)
                tot_sams=np.sum(stats_train[2])
                # tmp=np.sum([np.sum(self.lamda * ( th- thc ) ** 2) for th,thc in zip(self.latest_model,self.theta_c)])
                losses=[ n /tot_sams * loss for n,loss in zip(stats_train[2],stats_train[4])]
                tqdm.write('At round {} training loss: {}'.format(i,np.sum(losses)))
                loss_history.append(np.sum(losses))
            # choose M clients prop to data size, here need to choose all
            selected_clients = self.select_clients(i, num_clients=self.clients_per_round)
            # selected_clients=self.clients
            csolns = [] # buffer for receiving client solutions
            yy_ks = []
            #for c in tqdm(selected_clients, desc='Client: ', leave=False, ncols=120):
            for ci,c in enumerate(selected_clients):
                if ci==0:
                    grads=c.get_grads()
                    grads_sum0=[np.sum(x**2) for x in grads]
                    print('@HFfmaml line 41 sum grads:',np.sqrt(np.sum(grads_sum0)))
                # communicate the latest model
                c.model.receive_global_theta(self.latest_model)
                # solve minimization locally
                # nodes optimization
                soln,yy_k = c.solve_inner(num_epochs=self.num_epochs)
                # gather solutions from client
                csolns.append(soln)
                yy_ks.append(yy_k)
                # track communication cost
                # self.metrics.update(rnd=i, cid=c.id, stats=stats)
                # update model
            self.latest_model = self.aggregate(csolns,yy_ks)
            #print('@HFfmaml line48 latest_model',self.latest_model)
            # final test model
        stats = self.test()
        stats_train = self.train_error_and_loss()

        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)
        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds,
                                                                  np.sum(stats_train[3]) * 1.0 / np.sum(
                                                                      stats_train[2])))
        #print(len(stats_train))
        tqdm.write('At round {} training loss: {}'.format(self.num_rounds,np.mean(stats_train[4])))
        # save server model
        self.metrics.write()
        self.save()

        return loss_history
    ##@xinjiang
    def set_theta_c(self):
        theta_c=load_weights('weights.mat')
        #theta_c=self.client_model.get_params()
        #print('@HFmaml line 78 theta_c:', theta_c)
        self.theta_c=theta_c

from scipy import io
def load_weights(wPath='weights.mat'):
    params=io.loadmat(wPath)
    vars=list(params.values())[3:]
    vars=[np.squeeze(x) for x in vars]
    #print('@HFmaml line 85',vars)
    return vars