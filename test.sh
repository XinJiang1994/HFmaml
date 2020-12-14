d='cifar10'

echo $d

python3 main_HFfmaml.py --eval_every=9 --isTrain=True --dataset=cifar10 --model=cnn --clients_per_round=40 --num_rounds=100 --rho=0.7 --labmda=0 --sourceN=10 --R=0 --logdir=contrast_results100