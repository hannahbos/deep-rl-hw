import numpy as np
import pickle
import matplotlib.pyplot as plt

def read_data(filename):
    data= []
    with open(filename, 'rb') as fr:
        try:
            while True:
                data.append(pickle.load(fr))
        except EOFError:
            pass
    return np.asarray(data)

def compare_Q_and_double_Q():
    '''
        Shows performance of Q-learning algorithm with and without a second
        Q-network in the LunarLander environment.
    '''
    # set parameter for figure
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.figsize'] = (7.0, 2.0)
    plt.rcParams['ytick.labelsize'] = 7
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['font.size'] = 7
    plt.rcParams['legend.fontsize'] = 4.5
    plt.rcParams['lines.markersize'] = 2.0

    nx = 1
    ny = 3
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.5, hspace=0.5, top=0.95,
                        bottom=0.18, left=0.1, right=0.98)
    ax0 = plt.subplot2grid((nx,ny), (0,0))
    ax1 = plt.subplot2grid((nx,ny), (0,1))
    ax2 = plt.subplot2grid((nx,ny), (0,2))
    ax = [ax0, ax1, ax2]

    data_Q2 = [read_data('LunarLander_doubleQ_4563.pkl'),
               read_data('LunarLander_doubleQ_4564.pkl'),
               read_data('LunarLander_doubleQ_4565.pkl')]
    data_Q = [read_data('LunarLander_4563.pkl'),
              read_data('LunarLander_4564.pkl'),
              read_data('LunarLander_4565.pkl')]


    for i in range(3):
        ax[i].plot(data_Q2[i][0][0]/1e3, data_Q2[i][0][1], color='blue',
            label='mean episode reward, double Q')
        ax[i].plot(data_Q2[i][0][0]/1e3, data_Q2[i][0][2], color='darkblue',
            label='best mean episode reward, double Q')
        ax[i].plot(data_Q[i][0][0]/1e3, data_Q[i][0][1], color='red',
            label='mean episode reward')
        ax[i].plot(data_Q[i][0][0]/1e3, data_Q[i][0][2], color='darkred',
            label='best mean episode reward')
        ax[i].set_xticks([100,300,500])
        ax[i].set_xticklabels(['100k','300k','500k'])
        ax[i].set_xlabel('timesteps')
        ax[i].set_ylabel('return')

    ax[1].legend()

    plt.savefig('compare_Q_and_doubleQ.png')

def vary_target_update_freq():
    '''
        Shows performance of Q-learning algorithm with and without a second
        Q-network in the LunarLander environment for different values of the
        update frequency of the target network.
    '''
    # set parameter for figure
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.figsize'] = (3.0, 2.0)
    plt.rcParams['ytick.labelsize'] = 7
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['font.size'] = 7
    plt.rcParams['legend.fontsize'] = 4.5
    plt.rcParams['lines.markersize'] = 2.0

    nx = 1
    ny = 1
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.5, hspace=0.5, top=0.95,
                        bottom=0.18, left=0.18, right=0.98)
    ax0 = plt.subplot2grid((nx,ny), (0,0))

    env_name = 'LunarLander'
    seeds = ['4563', '4564', '4565']
    freqs = ['500', '1000', '3000', '4000', '10000']

    for j in range(6):
        filename = env_name
        if j>0:
            filename += '_doubleQ_freq' + freqs[j-1]
            label = 'f=' + freqs[j-1]
        else:
            label = 'single Q'
        filenames = [filename + '_' + seeds[k] + '.pkl' for k in range(3)]
        data = [read_data(filenames[k])[0] for k in range(3)]

        timesteps = np.asarray(data[0][0])/1e6
        rew = np.asarray([data[i][1] for i in range(3)])
        mean_rew = np.mean(rew, axis=0)

        ax0.plot(timesteps, mean_rew, label=label)
        ax0.set_xlabel('timesteps')
        ax0.set_ylabel('return')
        ax0.set_xticks([0.1, 0.3, 0.5])
        ax0.set_xticklabels(['100k', '300k', '500k'])
        ax0.set_ylim([-450,270])
        plt.legend()

    plt.savefig('vary_target_update_freq.png')

if __name__ == "__main__":
    compare_Q_and_double_Q()
    vary_target_update_freq()
