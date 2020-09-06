import numpy as np
import tensorflow as tf
from flearn.models.client_HFmaml import Client

class BaseFedarated(object):
    def __init__(self, params, learner, dataset):
        # transfer parameters to self
        for key, val in params.items(): setattr(self, key, val);
        _, _, train_data, test_data = dataset
        # create worker nodes
        tf.reset_default_graph()
        ##params['model_params'] --> num_classes
        params['w_i']=1
        self.learner=learner
        self.client_model =learner(params)
        self.latest_model = self.client_model.get_params()

        self.clients = self.setup_clients(dataset,params)
        print('{} Clients in Total'.format(len(self.clients)))


    def setup_clients(self, dataset,params):
        '''instantiates clients based on given train and test data directories
        Return:
            list of Clients
        client_model = learner(params['model_params'], self.opt1, self.opt2,, train_data, test_data,self.rho self.seed)
        '''
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]
        all_clients=[]
        total_sample_num=0
        w_is=[]
        for u, g in zip(users, groups):
            num_i=len(train_data[u]['y'])+len(test_data[u]['y'])
            w_is.append(num_i)
            total_sample_num+=num_i
        w_is=[x/total_sample_num for x in w_is]

        ## create clients
        for u, g,w_i in zip(users, groups,w_is):
            params['w_i']=w_i
            model = self.learner(params)
            all_clients.append(Client(u, g, train_data[u], test_data[u], model))
        return all_clients

    def train_error_and_loss(self):
        num_samples = []
        accs_train = []
        accs_test = []
        losses = []

        for c in self.clients:
            #c.set_params(self.latest_model)
            a_train,a_test, cl, ns= c.train_error_and_loss()
            accs_train.append(a_train)
            accs_test.append(a_test)
            num_samples.append(ns)
            losses.append(cl)
        
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]

        return ids, groups, num_samples, accs_test, losses,accs_train


    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        acc_test = []
        acc_train =[]
        #self.client_model.set_params(self.latest_model)
        for c in self.clients:
            c.set_params(self.latest_model)
            a_train,a_test, ns = c.test()
            acc_train.append(a_train * 1.0)
            acc_test.append(a_test)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, acc_test,acc_train

    def save(self):
        pass

    def select_clients(self, round, num_clients=20):
        '''selects num_clients clients weighted by number of samples from possible_clients
        
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''
        num_clients = min(num_clients, len(self.clients))
        np.random.seed(round)
        return np.random.choice(self.clients, num_clients, replace=False) #, p=pk)
