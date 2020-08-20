python3  -u main.py --dataset='mnist' --optimizer='fmaml'  \
            --learning_rate=0.01 --meta_rate=0.01\
            --num_rounds=40 --clients_per_round=80 \
            --mu=0.02 --eval_every=1 \
            --num_epochs=5 \
            --model='mclr'
