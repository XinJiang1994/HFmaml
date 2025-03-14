import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os
from flearn.models.client_maml import Client
from flearn.utils.model_utils import Metrics
from flearn.utils.tf_utils import process_grad

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
        #print(learner)
        #print(self.client_model)

        self.clients = self.setup_clients(dataset,params)
        print('{} Clients in Total'.format(len(self.clients)))
        #self.latest_model = self.client_model.get_params()
        # initialize system metrics
        self.metrics = Metrics(self.clients, params)


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
        accs = []
        losses = []

        for c in self.clients:
            #c.set_params(self.latest_model)
            acc, cl, ns= c.train_error_and_loss()
            accs.append(acc * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)
        
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]

        return ids, groups, num_samples, accs, losses


    def show_grads(self):  
        '''
        Return:
            gradients on all workers and the global gradient
        '''

        model_len = process_grad(self.latest_model).size
        global_grads = np.zeros(model_len)  

        intermediate_grads = []
        samples=[]

        #self.client_model.set_params(self.latest_model)
        for c in self.clients:
            c.set_params(self.latest_model)
            num_samples, client_grads = c.get_grads(self.latest_model) 
            samples.append(num_samples)
            global_grads = np.add(global_grads, client_grads * num_samples)
            intermediate_grads.append(client_grads)

        global_grads = global_grads * 1.0 / np.sum(np.asarray(samples)) 
        intermediate_grads.append(global_grads)

        return intermediate_grads
 
  
    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        #self.client_model.set_params(self.latest_model)
        for c in self.clients:
            c.set_params(self.latest_model)
            ct, ns = c.test()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, tot_correct

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


    def aggregate(self, wsolns):
        ###### platform aggregation operation #########
        ### This is a very important part Noted by XinJiang #######
        total_weight = 0.0
        base = [0]*len(wsolns[0][1])
        #print('@fedbase_maml line 139',wsolns)
        for (w, soln) in wsolns:  # w is the number of samples
            total_weight += w
            for i, v in enumerate(soln):
                base[i] += w*v.astype(np.float64)

        averaged_soln = [v / total_weight for v in base]


        return averaged_soln

