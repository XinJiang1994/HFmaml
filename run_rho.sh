
d='cifar10'
python3 main_HFfmaml.py \
          --eval_every=3 \
          --dataset=cifar10 \
          --model=cnn \
          --clients_per_round=40 \
          --num_rounds=100 \
          --isTrain=True \
          --rho=0.6 \
          --labmda=0 \
          --R=0 \
          --logdir=rho_test

python3 main_HFfmaml.py \
          --eval_every=3 \
          --dataset=$d \
          --model=cnn \
          --clients_per_round=40 \
          --num_rounds=100 \
          --isTrain=True \
          --rho=1 \
          --labmda=0 \
          --R=0 \
          --logdir=rho_test

python3 main_HFfmaml.py \
          --eval_every=3 \
          --dataset=$d \
          --model=cnn \
          --clients_per_round=40 \
          --num_rounds=100 \
          --isTrain=True \
          --rho=2 \
          --labmda=0 \
          --R=0 \
          --logdir=rho_test

python3 main_HFfmaml.py \
          --eval_every=3 \
          --dataset=$d \
          --model=cnn \
          --clients_per_round=40 \
          --num_rounds=100 \
          --isTrain=True \
          --rho=3 \
          --labmda=0 \
          --R=0 \
          --logdir=rho_test


d='Fmnist'
python3 main_HFfmaml.py \
          --eval_every=3 \
          --dataset=$d \
          --model=cnn \
          --clients_per_round=40 \
          --num_rounds=100 \
          --isTrain=True \
          --rho=0.3 \
          --labmda=0 \
          --R=0 \
          --logdir=rho_test

python3 main_HFfmaml.py \
          --eval_every=3 \
          --dataset=$d \
          --model=cnn \
          --clients_per_round=40 \
          --num_rounds=100 \
          --isTrain=True \
          --rho=0.6 \
          --labmda=0 \
          --R=0 \
          --logdir=rho_test

python3 main_HFfmaml.py \
          --eval_every=3 \
          --dataset=$d \
          --model=cnn \
          --clients_per_round=40 \
          --num_rounds=100 \
          --isTrain=True \
          --rho=1 \
          --labmda=0 \
          --R=0 \
          --logdir=rho_test

python3 main_HFfmaml.py \
          --eval_every=3 \
          --dataset=$d \
          --model=cnn \
          --clients_per_round=40 \
          --num_rounds=100 \
          --isTrain=True \
          --rho=2 \
          --labmda=0 \
          --R=0 \
          --logdir=rho_test