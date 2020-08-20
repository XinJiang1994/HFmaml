from flearn.utils.model_utils import read_data
import os
train_path = os.path.join('../data/synthetic_0.5_0.5/data', 'synthetic_0.5_0.5', 'data', 'train')
test_path = os.path.join('data',  'synthetic_0.5_0.5', 'data', 'test')
dataset = read_data(train_path, test_path)

print(dataset[4])