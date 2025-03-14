python3 data/DataDivision.py --dataset='cifar10'

for i in {1..10}
do

python3 main_HFfmaml.py \
        --eval_every=9 \
        --isTrain=True \
        --dataset=cifar10 \
        --model=cnn \
        --clients_per_round=40 \
        --num_rounds=100 \
        --rho=0.7 \
        --labmda=0 \
        --sourceN=10 \
        --R=$i \
        --logdir=contrast_results100

python3 main_fmaml.py --eval_every=3 \
        --dataset=cifar10 \
        --model=cnn_fmaml \
        --beta=0.005 \
        --num_epochs=1 \
        --num_rounds=100 \
        --R=$i \
        --logdir=contrast_results100

python3 main_fmaml.py --eval_every=3 \
        --dataset=cifar10 \
        --model=cnn_fmaml \
        --beta=0.005 \
        --num_epochs=5 \
        --num_rounds=100 \
        --R=$i \
        --logdir=contrast_results100

python3 main_fmaml.py --eval_every=3 \
        --dataset=cifar10 \
        --model=cnn_fmaml \
        --beta=0.005 \
        --num_epochs=10 \
        --num_rounds=100 \
        --R=$i \
        --logdir=contrast_results100

python3 Main_Fedrate.py \
        --eval_every=3 \
        --dataset=cifar10 \
        --model=cnn_fedavg \
        --num_rounds=100 \
        --clients_per_round=40 \
        --num_epochs=1 \
        --R=$i \
        --logdir=contrast_results100

python3 Main_Fedrate.py \
        --eval_every=3 \
        --dataset=cifar10 \
        --model=cnn_fedavg \
        --num_rounds=100 \
        --clients_per_round=40 \
        --num_epochs=5 \
        --R=$i \
        --logdir=contrast_results100

python3 Main_Fedrate.py \
        --eval_every=3 \
        --dataset=cifar10 \
        --model=cnn_fedavg \
        --num_rounds=100 \
        --clients_per_round=40 \
        --num_epochs=10 \
        --R=$i \
        --logdir=contrast_results100

done

