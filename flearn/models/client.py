import numpy as np

class Client(object):
    
    def __init__(self, id, group=None, train_data={'x':[],'y':[]}, eval_data={'x':[],'y':[]}, model=None):
        self.model = model
        self.id = id # integer
        self.group = group
        self.train_data = {k: np.array(v) for k,v in train_data.items()}
        self.eval_data = {k: np.array(v) for k,v in eval_data.items()}
        # print('@client line 13 eval_data:', self.eval_data['x'].shape)
        # print('@client line 13 train_data:',self.train_data['x'].shape)
        # print('@client line 13 train_data type:', type(self.train_data['x']))

        #self.data = self.train_data.update(self.eval_data)
        self.data = {key: (self.train_data[key], self.eval_data[key]) for key in self.train_data.keys() & self.eval_data}
        for k,v in self.data.items():
            self.data[k] = np.vstack(v)
        # print('@client line 13 data:', self.data['x'].shape)


        self.samples = len(self.data['y'])

        self.num_samples = len(self.train_data['y'])
        self.test_num = len(self.eval_data['y'])

    def set_params(self, model_params):
        '''set model parameters'''
        self.model.set_params(model_params)

    def get_params(self):
        '''get model parameters'''
        return self.model.get_params()

    def get_grads(self, model_len):
        '''get model gradient'''
        return self.model.get_gradients(self.data, model_len)
    #change train_data to data

    def solve_grad(self):
        '''get model gradient with cost'''
        bytes_w = self.model.size
        grads = self.model.get_gradients(self.data)
        comp = self.model.flops * self.samples
        bytes_r = self.model.size
        return ((self.samples, grads), (bytes_w, comp, bytes_r))
    #change num_samples to samples, train_data to data

    def solve_inner(self, num_epochs): #, batch_size=10):
        '''Solves local optimization problem
        
        Return:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes read: number of bytes received
            2: comp: number of FLOPs executed in training process
            2: bytes_write: number of bytes transmitted
        '''
        soln = self.model.solve_inner(self.data, num_epochs)#, batch_size)
        return (self.samples, soln)
    #change train.data to data, num_sam to sam, batch to len

    def test_inner(self, num_epochs):  # , batch_size=10):
        '''Solves local optimization problem

        Return:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes read: number of bytes received
            2: comp: number of FLOPs executed in training process
            2: bytes_write: number of bytes transmitted
        '''
        bytes_w = self.model.size
        batch_size = len(self.train_data['y'])
        soln, comp = self.model.solve_inner(self.train_data, num_epochs)  # , batch_size)
        bytes_r = self.model.size
        return soln
        #return (self.samples, soln), (bytes_w, comp, bytes_r)

    def test_train(self, data, num_epochs):
        batch_size = len(data['y'])
        soln, comp = self.model.solve_inner(data, num_epochs)#, batch_size)
        return  soln

    def train_error_and_loss(self):
        tot_correct, loss = self.model.test(self.data)
        return tot_correct, loss, self.samples


    def test(self):
        '''tests current model on local eval_data

        Return:
            tot_correct: total #correct predictions
            test_samples: int
        '''
        acc, loss = self.model.test(self.data)
        # print('data shape: ', len(self.data))
        # print('num_data: ', self.samples)
        return acc, self.samples

    def test_test(self):
        '''tests current model on local eval_data

        Return:
            tot_correct: total #correct predictions
            test_samples: int
        '''
        acc, loss,pred = self.model.test_test(self.eval_data)
        return  acc, loss, self.test_num,pred

    def test_zeroth(self):
        zero_loss = self.model.zeroth_loss(self.eval_data)
        return zero_loss


    def test_accuracy(self):
        tot_correct, loss = self.model.test(self.train_data)
        return tot_correct,loss, self.num_samples

    def final_test(self):
        tot_correct, loss, test_loss = self.model.final_test(self.train_data, self.eval_data)
        return tot_correct,loss, test_loss, self.num_samples

    def fast_adapt(self, num_epochs):
        #batch_size = len(self.train_data['y'])
        soln = self.model.fast_adapt(self.train_data, num_epochs)  # , batch_size)
        return soln

    #def test_loss(self):
    #    loss = self.model.test_loss(self.eval_data)
    #    return loss