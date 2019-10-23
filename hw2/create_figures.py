import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot import get_datasets

def get_max_returns(batches, rates):
    '''
        Reads training results for specified size of batches and learning rates
        and calculates the maximum return over iterations averaged accross three
        experiments.

        Arguments:
        batches: array of all batch sizes
        rates: array of all learning rates

        Returns:
        matrix of maximum averaged return with dimensions len(batches) x len(rates)
    '''
    sub_dirs = ['1/', '11/', '21']
    max_returns = np.zeros((len(batches),len(rates)))
    mean_after_max_returns = np.zeros((len(batches),len(rates)))
    nr_iterations = np.zeros((len(batches),len(rates)))
    for i,batch in enumerate(batches):
        for j,rate in enumerate(rates):
            logdir_base = 'data/IP_b' + str(batch) + '_r' + str(np.round(rate,3)) + '/'
            av_return = 0
            for sub in sub_dirs:
                data = get_datasets(logdir_base+sub)
                if isinstance(data, list):
                    data = pd.concat(data, ignore_index=True)
                # get average return for each iteration accross 3 experiments
                av_return += data['AverageReturn'].to_numpy()/3.
            max_returns[i][j] = np.max(av_return)
    return max_returns

def plot_heatmap():
    '''
        Plots a heatmap of the maximum return across iterations averaged over
        3 experiments for various batch sizes and learning rates.
        Data can be created by running run_hyperparameter_search_IP.sh
    '''
    # set parameter for figure
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.figsize'] = (3.0, 2.0)
    plt.rcParams['ytick.labelsize'] = 7
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['font.size'] = 7
    plt.rcParams['legend.fontsize'] = 7
    plt.rcParams['lines.markersize'] = 2.0

    nx = 1
    ny = 1
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.5, hspace=0.5, top=0.93,
                        bottom=0.2, left=0.2, right=0.8)
    ax = plt.subplot2grid((nx,ny), (0,0))

    # define batch size and learning rate arrays
    batches = np.arange(100, 210, 10)
    rates = np.arange(0.01, 0.05, 0.001)
    # get maximum returns across iterations
    max_returns = get_max_returns(batches, rates)
    vmin = np.min(max_returns)
    vmax = np.max(max_returns)

    # plot heatmap
    im1 = ax.pcolor(max_returns, vmin=vmin, vmax=vmax, cmap=plt.cm.Reds)
    ax.set_xticks([0.5 + 10*i for i in range(len(rates[::10]))])
    ax.set_xticklabels(np.round(rates[::10],3))
    ax.set_yticks([0.5 + 2*i for i in range(len(batches[::2]))])
    ax.set_yticklabels(batches[::2])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    cbar_ax = fig.add_axes([box.x0 + box.width + 0.005*box.width,
                                box.y0, 0.02, box.height])
    ax.set_xlabel('learning rate')
    ax.set_ylabel('batch size')
    cb = fig.colorbar(im1, cax=cbar_ax)

    cb.set_label('max. averaged reward')

    plt.savefig('hyperparameter_search_IP.png')


if __name__ == "__main__":
    plot_heatmap()
