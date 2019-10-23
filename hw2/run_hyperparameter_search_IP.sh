#!/bin/bash

# run training sessions for grid search in Problem 5
for l in $(seq 0.01 0.001 0.05)
do
  for b in $(seq 110 10 200)
  do
    echo $b $l
    python3 train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b $b -lr $l --exp_name IP_b"$b"_r"$l"
    # rename folder
    mv data/IP_b"$b"_r"$l"* data/IP_b"$b"_r"$l"
  done
done
