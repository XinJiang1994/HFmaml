import numpy as np
from tqdm import trange, tqdm
from flearn.utils.model_utils import load_weights
from .fedbase_HFmaml import BaseFedarated
from flearn.models.client_HFmaml import Client

class Server(BaseFedarated):
    def __init__(self, params, learner, datasets,theta_c_path,test_user):

        print('Using HFmaml to Train')
        self.theta_c_path=theta_c_path
        self.lamda=params['labmda']
        self.test_user=test_user
        # self.learner=learner #super class has already set self.learner
        self.params=params
        self.datasets_data=datasets #注意这里的dataset_data是真的dataset，还有一个self.dataset实际是dataset name
        _, _, self.train_data, self.test_data = datasets
        super(Server, self).__init__(params, learner, datasets)
        ### @xinjiang set theta_c ### end
        self.set_theta_c()

    def train(self):
        print('Training with {} workers ---'.format(self.clients_per_round))
        ## num_rounds is k
        ## num_epochs should set 1 in HFfmaml
        loss_history=[]
        acc_history=[]
        for i in trange(self.num_rounds, desc='Round: ', ncols=120):
            # test model
            if i % self.eval_every == 0:
                # evalute source node
                for c in self.clients:
                    c.set_params(self.latest_model)

                # print('::::::::::::::::::::phy:',np.sum([np.linalg.norm(x) for x in self.clients[0].get_features_train()]))
                stats_train = self.train_error_and_loss()
                tot_sams=np.sum(stats_train[2])


                losses=[ n / tot_sams * loss for n,loss in zip(stats_train[2],stats_train[4])]
                accs = [n / tot_sams * acc for n, acc in zip(stats_train[2], stats_train[3])]
                accs_train = [n / tot_sams * acc for n, acc in zip(stats_train[2], stats_train[5])]

                # evalute target node
                acc_target='None'
                acc_target=target_test2(self.test_user,self.learner,self.datasets_data,self.params,self.latest_model)

                # print evalution results
                tqdm.write('At round {} training loss: {}; acc_train:{}; acc_test:{}, target acc:{}'.format(i,np.sum(losses),np.sum(accs_train),np.sum(accs),acc_target))
                loss_history.append(np.sum(losses))
                acc_history.append(acc_target)
            selected_clients=self.clients
            # selected_clients = self.select_clients(i, num_clients=self.clients_per_round)
            solns = [] # buffer for receiving client solutions
            yy_ks = []
            #for c in tqdm(selected_clients, desc='Client: ', leave=False, ncols=120):
            for ci,c in enumerate(selected_clients):
                # communicate the latest model
                c.model.receive_global_theta(self.latest_model)
                # solve minimization locally
                # nodes optimization
                soln,yy_k = c.solve_inner(num_epochs=self.num_epochs)
                # gather solutions from client
                solns.append(soln)
                yy_ks.append(yy_k)
                # track communication cost
                # update model
            self.latest_model = self.aggregate(solns,yy_ks)
        stats_train = self.train_error_and_loss()

        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds,
                                                                  np.sum(stats_train[3]) * 1.0 / np.sum(
                                                                      stats_train[2])))
        tqdm.write('At round {} training loss: {}'.format(self.num_rounds,np.mean(stats_train[4])))
        # self.save()

        return loss_history,acc_history
    ##@xinjiang
    def set_theta_c(self):
        if self.labmda == 0:
            theta_c=self.client_model.get_params()
        else:
            print('#### Loading theta_c...')
            theta_c = load_weights(self.theta_c_path)
            # model_param=self.client_model.get_params()
            # theta_c=[np.random.normal(0.01, 0.5, p.shape) for p in model_param]
            # print('@HFmaml line 78 theta_c:', theta_c)
        self.theta_c=theta_c

    def aggregate(self, solns,yy_ks):

        l_th_c = [2*self.labmda*t for t in self.theta_c]

        # solns is n个node的param list

        n=len(solns) # totally n nodes
        m=len(solns[0]) #param list的length
        # all rhos are the same, so we can just use self.rho
        sum_rho = self.rho * n
        sum_yy_theta=[]

        for j in range(m):
            tmp_v= np.zeros_like(solns[0][j])
            for i in range(n):
                tmp_v += (yy_ks[i][j]+self.rho * solns[i][j])
            sum_yy_theta.append(tmp_v)
        theta_kp1=[(ltc+syt)/(2*self.labmda + sum_rho) for ltc,syt in zip(l_th_c,sum_yy_theta)]
        return theta_kp1


def target_test2(test_user,learner,dataset,options,weight):
    accs=dict()
    num_test=dict()
    for i,user in enumerate(test_user):
        accs[i],num_test[i]=final_test(learner=learner, train_data=dataset[2][user], test_data=dataset[3][user],
                params=options, user_name=user, weight= weight)
    accs=list(accs.values())
    num_test=list(num_test.values())
    acc_test = [a * n/np.sum(num_test) for a, n in zip(accs, num_test)]
    return np.sum(acc_test)

def final_test(learner, train_data, test_data, params, user_name, weight):
    # print('HFmaml test')
    params['w_i']=1
    client_model = learner(params)  # changed remove star
    test_client = Client(user_name, [], train_data, test_data, client_model)
    test_client.set_params(weight)
    acc,numsam = test_client.target_acc_while_train()
    return acc,numsam