import numpy as np
import matplotlib.pyplot as plt

def get_nr_params_ff_base(h, params):
    dim_ob, dim_ac, size, gru_size, n_layers = params
    l1 = dim_ob*h*size
    b1 = size
    l2 = n_layers*size*size
    b2 = n_layers*size
    l3 = size*gru_size
    b3 = gru_size
    return l1+b1+l2+b2+l3+b3

def get_nr_params_policy_ff(h, params):
    dim_ob, dim_ac, size, gru_size, n_layers = params
    l4 = gru_size*dim_ac
    b4 = dim_ac
    return get_nr_params_ff_base(h, params)+l4+b4

def get_nr_params_critic_ff(h, params):
    dim_ob, dim_ac, size, gru_size, n_layers = params
    l4 = gru_size*1
    b4 = 1
    return get_nr_params_ff_base(h, params)+l4+b4

def get_nr_params_rec_base(params):
    dim_ob, dim_ac, size, gru_size, n_layers = params
    l1 = dim_ob*size
    b1 = size
    l2 = n_layers*size*size
    b2 = n_layers*size
    l3 = size*dim_ob
    b3 = dim_ob
    l4 = (gru_size+dim_ob)*2*gru_size
    b4 = 2*gru_size
    l5 = (gru_size+dim_ob)*gru_size
    b5 = gru_size
    return l1+b1+l2+b2+l3+b3+l4+b4+l5+b5

def get_nr_params_policy_rec(params):
    dim_ob, dim_ac, size, gru_size, n_layers = params
    l6 = gru_size*dim_ac
    b6 = dim_ac
    return get_nr_params_rec_base(params)+l6+b6

def get_nr_params_critic_rec(params):
    dim_ob, dim_ac, size, gru_size, n_layers = params
    l6 = gru_size*1
    b6 = 1
    return get_nr_params_rec_base(params)+l6+b6

def plot_nr_of_parameter():
    '''
        Visualizes number of parameter in feedforward and recurrent network.
    '''
    # set parameter for figure
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.figsize'] = (2.5, 2.0)
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['font.size'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['lines.markersize'] = 2.0

    nx = 1
    ny = 1
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.5, hspace=0.5, top=0.95,
                        bottom=0.2, left=0.22, right=0.95)
    ax0 = plt.subplot2grid((nx,ny), (0,0))

    dim_ob = 6
    dim_ac = 4
    size = 64
    gru_size = 32
    n_layers = 0
    params = [dim_ob, dim_ac, size, gru_size, n_layers]

    h = np.arange(120)
    params_ff = get_nr_params_policy_ff(h, params)+get_nr_params_critic_ff(h, params)
    params_rec = get_nr_params_policy_rec(params)+get_nr_params_critic_rec(params)

    ax0.plot(h, params_rec*np.ones_like(h)/1000, color='k', label='rec')
    ax0.plot(h, params_ff/1000, color='gray', label='ff')
    ax0.set_xlabel('history length')
    ax0.set_ylabel('number of parameter')
    ax0.set_yticks([20,40,60,80])
    ax0.set_yticklabels(['20k','40k','60k','80k'])
    ax0.legend()

    plt.savefig('nr_of_parameter.png')

if __name__ == "__main__":
    plot_nr_of_parameter()
