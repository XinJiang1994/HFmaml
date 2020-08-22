import numpy as np


class Client(object):

    def __init__(self, id, group=None, train_data={'x':[],'y':[]}, eval_data={'x':[],'y':[]}, model=None):
        self.model = model
        self.id = id #integer
        self.group = group
        self.train_data = {k: np.array(v) for k,v in train_data.items()}
        self.eval_data = {k: np.array(v) for k,v in eval_data.items()}
        self.train_samples = len(self.train_data['y'])
        self.num_samples = len(self.eval_data['y']) # why this could be zero, this leads to the problem, this is just one datapoint
        # need to check the definition of num_samples


    def set_params(self, model_params):
        '''set model parameters'''
        self.model.set_params(model_params)


    def get_params(self):
        '''get model parameters'''
        return self.model.get_params()


    def get_grads(self, model_len):
        '''get model gradient'''
        return self.model.get_gradients(self.train_data, self.eval_data, model_len)
        # why is train_data?


    def solve_grad(self):
        '''get model gradient with cost'''
        bytes_w = self.model.size
        grads = self.model.get_gradients(self.train_data)
        comp = self.model.flops * self.train_samples
        bytes_r = self.model.size
        return ((self.train_samples, grads), (bytes_w, comp, bytes_r))


    def solve_inner(self, num_epochs):
        '''solve local optimization problem

        Return:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes_read: number of bytes received
            2: comp: number of flops executed in training process
            2: bytes_write: number of bytes transmitted
        '''

        soln= self.model.solve_inner(self.train_data, self.eval_data, num_epochs)
        return (self.num_samples, soln)
        # change this, since for clients, two steps needed, not just solve_inner
        # add eval_data, but how to deal with epoch and batch
        # use same epoch, add another batch_size as input
        # inner_opt, learning_rate in fedbase


    # loss part may need to change
    # training error is testing error, do not need to test again
    def train_error_and_loss(self):
        tot_correct, loss = self.model.test(self.train_data, self.eval_data)
        return tot_correct, loss, self.num_samples


    def test(self):
        '''return: tot_correct: total # correct predictions'''
        acc, loss = self.model.test(self.train_data, self.eval_data)
        return acc, self.num_samples

    def test_test(self):
        '''tests current model on local eval_data

        Return:
            tot_correct: total #correct predictions
            test_samples: int
        '''
        acc, loss,pred = self.model.test_test(self.eval_data)
        return acc, loss, self.num_samples

    def test_zeroth(self):
        zero_loss = self.model.zeroth_loss(self.eval_data)
        return zero_loss

    def test_train(self):
        batch_size = len(self.train_data['y'])
        soln = self.model.test_train(self.train_data)  # , batch_size)
        return soln

    def fast_adapt(self, num_epochs):
        #batch_size = len(self.train_data['y'])
        soln= self.model.fast_adapt(self.train_data, num_epochs)  # , batch_size)
        return soln

    # all these functions have been defined for the model, so directly
    # use them here
