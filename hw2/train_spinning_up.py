from spinup import vpg, trpo, ppo
import tensorflow as tf
import gym
import sys

env_fn = lambda : gym.make('LunarLander-v2')

ac_kwargs = dict(hidden_sizes=[64,64], activation=tf.nn.tanh)

seed = 1
epochs = 100 # number of policy updates
steps_per_epoch = 40000 # comparable to batch size in hw2
# string specifying the polic gradient algorithm: vpg, ppo, trpo
algorithm = sys.argv[1]

# train with vanilla policy gradient
if algorithm == 'vpg':
    lam = sys.argv[2]
    exp_name = 'll_vpg_seed' + str(seed) + '_epochs' + str(epochs) + '_lam' + lam
    logger_kwargs = dict(output_dir='data_spinning_up/' + exp_name + '/', exp_name=exp_name)
    vpg(env_fn=env_fn, ac_kwargs=ac_kwargs, max_ep_len=1000, gamma=0.99, seed=seed,
        steps_per_epoch=steps_per_epoch, pi_lr=0.005, vf_lr=0.005, epochs=epochs,
        logger_kwargs=logger_kwargs, lam=float(lam))

# train with PPO
if algorithm == 'ppo':
    clip_ratio = sys.argv[2]
    target_kl = sys.argv[3]
    exp_name = 'll_ppo_seed' + str(seed) + '_epochs' + str(epochs)
    exp_name += '_cr' + clip_ratio + '_tk' + target_kl
    logger_kwargs = dict(output_dir='data_spinning_up/' + exp_name + '/', exp_name=exp_name)
    ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, max_ep_len=1000, gamma=0.99, seed=seed,
        steps_per_epoch=steps_per_epoch, pi_lr=0.005, vf_lr=0.005, epochs=epochs,
        logger_kwargs=logger_kwargs, clip_ratio=float(clip_ratio),
        target_kl=float(target_kl))

# train with TRPO
if algorithm == 'trpo':
    delta = sys.argv[2]
    backtrack_coef = sys.argv[3]
    exp_name = 'll_trpo_seed' + str(seed) + '_epochs' + str(epochs)
    exp_name += '_delta' + delta + '_bc' + backtrack_coef
    logger_kwargs = dict(output_dir='data_spinning_up/' + exp_name + '/', exp_name=exp_name)
    trpo(env_fn=env_fn, ac_kwargs=ac_kwargs, max_ep_len=1000, gamma=0.99, seed=seed,
         steps_per_epoch=steps_per_epoch, vf_lr=0.005, epochs=epochs,
         logger_kwargs=logger_kwargs, backtrack_coeff=float(backtrack_coef),
         delta=float(delta))
