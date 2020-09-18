ds=cifar100
logdir=contrast_results_cifar100_r100

clients_per_round=80

python3 data/DataDivision.py --dataset='cifar100'

for i in {1..10}
do

python3 main_HFfmaml.py \
        --eval_every=3 \
        --isTrain=True \
        --dataset=$ds \
        --model=cnn \
        --clients_per_round=$clients_per_round \
        --num_rounds=100 \
        --rho=0.7 \
        --labmda=0 \
        --sourceN=10 \
        --R=$i \
        --logdir=$logdir

python3 main_fmaml.py --eval_every=3 \
        --dataset=$ds \
        --model=cnn_fmaml \
        --beta=0.001 \
        --num_epochs=1 \
        --num_rounds=100 \
        --clients_per_round=$clients_per_round  \
        --R=$i \
        --logdir=$logdir

python3 main_fmaml.py --eval_every=3 \
        --dataset=$ds \
        --model=cnn_fmaml \
        --beta=0.001 \
        --num_epochs=5 \
        --num_rounds=100 \
         --clients_per_round=$clients_per_round  \
        --R=$i \
        --logdir=$logdir

python3 main_fmaml.py --eval_every=3 \
        --dataset=$ds \
        --model=cnn_fmaml \
        --beta=0.001 \
        --num_epochs=10 \
        --num_rounds=100 \
         --clients_per_round=$clients_per_round  \
        --R=$i \
        --logdir=$logdir

python3 Main_Fedrate.py \
        --eval_every=3 \
        --dataset=$ds \
        --model=cnn_fedavg \
        --num_rounds=100 \
        --clients_per_round=$clients_per_round  \
        --num_epochs=1 \
        --R=$i \
        --logdir=$logdir

python3 Main_Fedrate.py \
        --eval_every=3 \
        --dataset=$ds \
        --model=cnn_fedavg \
        --num_rounds=100 \
        --clients_per_round=$clients_per_round  \
        --num_epochs=5 \
        --R=$i \
        --logdir=$logdir

python3 Main_Fedrate.py \
        --eval_every=3 \
        --dataset=$ds \
        --model=cnn_fedavg \
        --num_rounds=100 \
        --clients_per_round=$clients_per_round  \
        --num_epochs=10 \
        --R=$i \
        --logdir=$logdir

done

