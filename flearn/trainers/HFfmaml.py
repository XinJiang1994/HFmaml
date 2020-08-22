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
        self.set_theta_c(params)

    def train(self):
        print('Training with {} workers ---'.format(self.clients_per_round))
        ## num_rounds is k
        ## num_epochs should set 1 in HFfmaml
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
            # choose M clients prop to data size, here need to choose all
            selected_clients = self.select_clients(i, num_clients=self.clients_per_round)
            # selected_clients=self.clients
            csolns = [] # buffer for receiving client solutions
            yy_ks = []
            #for c in tqdm(selected_clients, desc='Client: ', leave=False, ncols=120):
            for ci,c in enumerate(selected_clients):
                # communicate the latest model
                c.model.receive_global_theta(self.latest_model)
                # solve minimization locally
                # nodes optimization
                soln, stats,yy_k = c.solve_inner(num_epochs=self.num_epochs)
                # gather solutions from client
                csolns.append(soln)
                yy_ks.append(yy_k)
                # track communication cost
                self.metrics.update(rnd=i, cid=c.id, stats=stats)
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
    ##@xinjiang
    def set_theta_c(self,params):
        theta_c = []
        for it, par in enumerate(self.client_model.get_params()):
            seed = 132 + params['seed'] + it
            ## the shape of params might different, for example: w--(784，10), b--(10,)
            len_c = 1
            for s in par.shape:
                len_c *= s
            np.random.seed(seed)
            th_c_flat = np.random.rand(len_c)
            th_c = th_c_flat.reshape(par.shape)
            # print('th_c.shape',th_c.shape)
            # th_c = np.zeros_like(par)
            theta_c.append(th_c)

        self.theta_c=theta_c
    ##@xinjiang
    # Overload the parent function which is in fedbase_maml.py
    # note: solns means solutions which are the trainable params in the model.
    def aggregate(self, wsolns,yy_ks):
        solns=[]
        for w,slon in wsolns:
            solns.append(slon)
        l_th_c=[2*self.labmda*t for t in self.theta_c]

        n=len(solns) # totally n nodes
        m=len(solns[0]) #[w,b]
        # all rhos are the same, so we can just use self.rho
        sum_rho = self.rho * n
        sum_yy_theta=[]
        for j in range(m):
            tmp_v= np.zeros_like(solns[0][j])
            for i in range(n):
                tmp_v += (yy_ks[i][j]+self.rho * solns[i][j])
            sum_yy_theta.append(tmp_v)
        theta_kp1=[]
        for ltc,syt in zip(l_th_c,sum_yy_theta):
            theta_kp1.append((ltc+syt)/(2*self.labmda + sum_rho))
        return theta_kp1

