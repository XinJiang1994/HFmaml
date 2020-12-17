#gennerate test dataset
python3 data/DataDivisionFor_Forget_test.py --dataset=cifar10
python3 data/DataDivisionFor_Forget_test.py --dataset=cifar100
python3 data/DataDivisionFor_Forget_test.py --dataset=Fmnist


#cifar10

python3 main_HFfmaml.py --eval_every=3 --dataset=cifar10 --model=cnn --clients_per_round=40 --num_rounds=100  --sourceN=10 --R=0 --logdir=contrast_results100 --labmda=0 --rho=0.7 --isTrain=True --pretrain=True


python3 main_HFfmaml.py --eval_every=3 --dataset=cifar10 --model=cnn --clients_per_round=40 --num_rounds=100  --sourceN=10 --R=0 --logdir=contrast_results100 --labmda=1 --rho=0.7 --isTrain=True --transfer=True

python3 main_fmaml.py --eval_every=3 --dataset=cifar10 --model=cnn_fmaml --beta=0.001 --num_epochs=1 --num_rounds=100 --R=0 --logdir=contrast_results100 --transfer=True

python3 Main_Fedrate.py --eval_every=3 --dataset=cifar10 --model=cnn_fedavg --num_rounds=100 --clients_per_round=40 --num_epochs=1 --R=0 --logdir=contrast_results_cifar100_r100 --transfer=True

#cifar100
python3 main_HFfmaml.py --eval_every=3 --dataset=cifar100 --model=cnn --clients_per_round=40 --num_rounds=100  --sourceN=10 --R=0 --logdir=contrast_results100 --labmda=0 --rho=1 --isTrain=True --pretrain=True

python3 main_HFfmaml.py --eval_every=3 --dataset=cifar100 --model=cnn --clients_per_round=40 --num_rounds=100  --sourceN=10 --R=0 --logdir=contrast_results100 --labmda=1 --rho=1.5 --isTrain=True --transfer=True

python3 main_fmaml.py --eval_every=3 --dataset=cifar100 --model=cnn_fmaml --beta=0.001 --num_epochs=10 --num_rounds=100 --R=0 --logdir=contrast_results100 --transfer=True

python3 Main_Fedrate.py --eval_every=3 --dataset=cifar100 --model=cnn_fedavg --num_rounds=100 --clients_per_round=$clients_per_round --num_epochs=10 --R=0 --logdir=contrast_results_cifar100_r100 --transfer=True


#Fmnist
python3 main_HFfmaml.py --eval_every=3 --dataset=Fmnist --model=cnn --clients_per_round=40 --num_rounds=100  --sourceN=10 --R=0.35 --logdir=contrast_results100 --labmda=0 --rho=0.7 --isTrain=True --pretrain=True

python3 main_HFfmaml.py --eval_every=3 --isTrain=True --dataset=Fmnist --model=cnn --clients_per_round=40 --num_rounds=100  --sourceN=10 --R=0 --logdir=contrast_results100 --labmda=0.5 --rho=0.35 --transfer=True

python3 main_fmaml.py --eval_every=3 --dataset=Fmnist --model=cnn_fmaml --beta=0.005 --num_epochs=1 --num_rounds=100 --R=0 --logdir=contrast_results100 --transfer=True

python3 Main_Fedrate.py --eval_every=3 --dataset=Fmnist --model=cnn_fedavg --num_rounds=100 --clients_per_round=40 --num_epochs=1 --R=0 --logdir=contrast_results_cifar100_r100 --transfer=True