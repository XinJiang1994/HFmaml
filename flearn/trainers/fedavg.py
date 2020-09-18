import numpy as np
from tqdm import trange, tqdm

from .fedbase import BaseFedarated
from flearn.models.client import Client

class Server(BaseFedarated):
    def __init__(self, params, learner, dataset,test_user):
        print('Using Federated Average to Train')
        #self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        self.opt1 = params['alpha']
        self.test_user = test_user
        # self.learner=learner #super class has already set self.learner
        self.params = params
        self.datasets_data = dataset  # 注意这里的dataset_data是真的dataset，还有一个self.dataset实际是dataset name
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        '''Train using Federated Averaging'''
        print('Training with {} workers ---'.format(self.clients_per_round))
        loss_history=[]
        acc_history = []
        for i in trange(self.num_rounds, desc='Round: ', ncols=120):
            # test model
            if i % self.eval_every == 0:
                for c in self.clients:
                    # communicate the latest model
                    c.set_params(self.latest_model)
                stats = self.test()
                stats_train = self.train_error_and_loss()
                self.metrics.accuracies.append(stats)
                self.metrics.train_accuracies.append(stats_train)
                tot_sams = np.sum(stats_train[2])
                # tmp=np.sum([np.sum(self.lamda * ( th- thc ) ** 2) for th,thc in zip(self.latest_model,self.theta_c)])
                losses=[ n / tot_sams * loss for n,loss in zip(stats_train[2],stats_train[4])]
                accs = [n / tot_sams * acc for n, acc in zip(stats_train[2], stats_train[3])]

                acc_target='XXX'
                acc_target = target_test2(self.test_user, self.learner, self.datasets_data, self.params,
                                          self.latest_model)

                # print('@HFmaml line32 stats_train:',stats_train[2:])
                tqdm.write('At round {} training loss: {}; acc:{}, target acc: {}'.format(i,np.sum(losses),np.sum(accs),acc_target))
                loss_history.append(np.sum(losses))
                acc_history.append(acc_target)
                # tqdm.write('At round {} accuracy: {}'.format(i, np.sum(stats[3])*1.0/np.sum(stats[2])))
                # tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))

                # save server model
                self.metrics.write()
                self.save()

            # choose K clients prop to data size
            selected_clients = self.select_clients(i, num_clients=self.clients_per_round)
            # print('Training with {} workers ---'.format(len(selected_clients)))

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
        return loss_history,acc_history


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