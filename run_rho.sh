for i in {1..10}
do

d='cifar10'
#python3 main_HFfmaml.py \
#          --eval_every=3 \
#          --dataset=cifar10 \
#          --model=cnn \
#          --clients_per_round=40 \
#          --num_rounds=100 \
#          --isTrain=True \
#          --rho=0.7 \
#          --labmda=0 \
#          --R=$i \
#          --logdir=rho_test
#
#python3 main_HFfmaml.py \
#          --eval_every=3 \
#          --dataset=$d \
#          --model=cnn \
#          --clients_per_round=40 \
#          --num_rounds=100 \
#          --isTrain=True \
#          --rho=2 \
#          --labmda=0 \
#          --R=$i \
#          --logdir=rho_test

python3 main_HFfmaml.py \
          --eval_every=3 \
          --dataset=$d \
          --model=cnn \
          --clients_per_round=40 \
          --num_rounds=100 \
          --isTrain=True \
          --rho=5 \
          --labmda=0 \
          --R=$i \
          --logdir=rho_test

#python3 main_HFfmaml.py \
#          --eval_every=3 \
#          --dataset=$d \
#          --model=cnn \
#          --clients_per_round=40 \
#          --num_rounds=100 \
#          --isTrain=True \
#          --rho=10 \
#          --labmda=0 \
#          --R=$i \
#          --logdir=rho_test

done