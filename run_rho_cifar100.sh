python3 data/DataDivision.py --dataset='cifar100'

for i in {1..1}
do

d='cifar100'
logdir='rho_test100'
clients_per_round=80

python3 main_HFfmaml.py \
          --eval_every=3 \
          --dataset=cifar10 \
          --model=cnn \
          --clients_per_round=$clients_per_round \
          --num_rounds=100 \
          --isTrain=True \
          --rho=0.7 \
          --labmda=0 \
          --R=$i \
          --logdir=$logdir

python3 main_HFfmaml.py \
          --eval_every=3 \
          --dataset=$d \
          --model=cnn \
          --clients_per_round=$clients_per_round \
          --num_rounds=100 \
          --isTrain=True \
          --rho=1 \
          --labmda=0 \
          --R=$i \
          --logdir=$logdir

python3 main_HFfmaml.py \
          --eval_every=3 \
          --dataset=$d \
          --model=cnn \
          --clients_per_round=$clients_per_round \
          --num_rounds=100 \
          --isTrain=True \
          --rho=2 \
          --labmda=0 \
          --R=$i \
          --logdir=rho_test

python3 main_HFfmaml.py \
          --eval_every=3 \
          --dataset=$d \
          --model=cnn \
          --clients_per_round=$clients_per_round \
          --num_rounds=100 \
          --isTrain=True \
          --rho=5 \
          --labmda=0 \
          --R=$i \
          --logdir=$logdir

python3 main_HFfmaml.py \
          --eval_every=3 \
          --dataset=$d \
          --model=cnn \
          --clients_per_round=$clients_per_round \
          --num_rounds=100 \
          --isTrain=True \
          --rho=10 \
          --labmda=0 \
          --R=$i \
          --logdir=$logdir

done