from flearn.models.ThetaC.cnn_model import Model
import numpy as np

def main():
    model = Model(model_name='cifar10',params={'num_classes':10})
    model.train(bath_size=45,num_epoch=100)
    model.save_thetaC()


def test():
    data_path='/root/TC174611125/fmaml/fmaml_mac/data/cifar10/center_data.npz'
    data = np.load(data_path)
    print(data['X'])

if __name__=="__main__":
    main()