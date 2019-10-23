#!/bin/bash

# create data for LunarLander environment with spinningup implementation of
# policy gradients algorithm

for lam in 0.97 1
do
  python3 train_spinning_up.py vpg $lam
done

for clip_ratio in 0.1 0.2 0.3
do
  for target_kl in 0.01 0.03 0.05
  do
    python3 train_spinning_up.py ppo $clip_ratio $target_kl
  done
done

for delta in 0.01 0.03 0.05
do
  for backtrack_coef in 0.25 0.5 0.75
  do
    python3 train_spinning_up.py trpo $delta $backtrack_coef
  done
done
