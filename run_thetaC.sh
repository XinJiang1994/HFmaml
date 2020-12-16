python3 data/DataDivisionForThetaC.py

 python3 main_HFfmaml.py \
          --eval_every=3 \
          --dataset=cifar10 \
          --model=cnn \
          --clients_per_round=40 \
          --num_rounds=100 \
          --isTrain=True \
          --rho=0.7 \
          --labmda=0 \
          --R=0 \
          --logdir=theta_c_test_results \
          --pretrain=True


for i in {1..10}
do
 python3 main_HFfmaml.py \
          --eval_every=3 \
          --dataset=cifar10 \
          --model=cnn \
          --clients_per_round=40 \
          --num_rounds=100 \
          --isTrain=True \
          --rho=10 \
          --labmda=3 \
          --R=$i \
          --logdir=theta_c_test_results


 python3 main_HFfmaml.py \
          --eval_every=3 \
          --dataset=cifar10 \
          --model=cnn \
          --clients_per_round=40 \
          --num_rounds=100 \
          --isTrain=True \
          --rho=10 \
          --labmda=1 \
          --R=$i \
          --logdir=theta_c_test_results

 python3 main_HFfmaml.py \
          --eval_every=3 \
          --dataset=cifar10 \
          --model=cnn \
          --clients_per_round=40 \
          --num_rounds=100 \
          --isTrain=True \
          --rho=10 \
          --labmda=5 \
          --R=$i \
          --logdir=theta_c_test_results

 python3 main_HFfmaml.py \
          --eval_every=3 \
          --dataset=cifar10 \
          --model=cnn \
          --clients_per_round=40 \
          --num_rounds=100 \
          --isTrain=True \
          --rho=10 \
          --labmda=10 \
          --R=$i \
          --logdir=theta_c_test_results

 python3 main_HFfmaml.py \
          --eval_every=3 \
          --dataset=cifar10 \
          --model=cnn \
          --clients_per_round=40 \
          --num_rounds=100 \
          --isTrain=True \
          --rho=10 \
          --labmda=20 \
          --R=$i \
          --logdir=theta_c_test_results

 python3 main_HFfmaml.py \
          --eval_every=3 \
          --dataset=cifar10 \
          --model=cnn \
          --clients_per_round=40 \
          --num_rounds=100 \
          --isTrain=True \
          --rho=10 \
          --labmda=50 \
          --R=$i \
          --logdir=theta_c_test_results


python3 main_HFfmaml.py \
          --eval_every=3 \
          --dataset=cifar10 \
          --model=cnn \
          --clients_per_round=40 \
          --num_rounds=100 \
#          --isTrain=True \
          --rho=10 \
          --labmda=30 \
          --R=$i \
          --logdir=theta_c_test_results
done