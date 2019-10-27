import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from scipy.signal import welch
import csv

import h5py_wrapper_python3.wrapper as h5

def read_data(filename):
    data= []
    with open(filename, 'rb') as fr:
        try:
            while True:
                data.append(pickle.load(fr))
        except EOFError:
            pass
    return np.asarray(data)

def plots_part1():
    '''
        Visualizes differences between learned and real dynamics.
    '''
    # set parameter for figure
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.figsize'] = (4.0, 2.0)
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['font.size'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['lines.markersize'] = 2.0

    nx = 1
    ny = 2
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.5, hspace=0.5, top=0.85,
                        bottom=0.15, left=0.1, right=0.95)
    ax0 = plt.subplot2grid((nx,ny), (0,0))
    ax1 = plt.subplot2grid((nx,ny), (0,1))
    ax = [ax0, ax1]

    states = h5.load_h5('results_hw1.h5', '/0/states')
    pred_states = h5.load_h5('results_hw1.h5', '/0/pred_states')

    for i, j in enumerate([11,17]):
        ax[i].plot(states[:,j], 'k', label='original')
        ax[i].plot(pred_states[:,j], 'r', label='model')
        ax[i].set_xlim([0,100])
        ax[i].set_title('state ' + str(j))
    ax[1].legend()

    plt.savefig('plots_part1.png')

def plots_part3():
    '''
        Visualizes returns over iterations.
    '''
    # set parameter for figure
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.figsize'] = (3.0, 2.0)
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['font.size'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['lines.markersize'] = 2.0

    nx = 1
    ny = 1
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.5, hspace=0.5, top=0.85,
                        bottom=0.2, left=0.22, right=0.95)
    ax0 = plt.subplot2grid((nx,ny), (0,0))
    ax = [ax0]

    average_return = []
    std_return = []
    with open('data/HalfCheetah_q3/log.csv', 'rt') as csvfile:
        data = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for i,row in enumerate(data):
            if i>0:
                _, _, ar, sr, _, _, _, _ = row[0].split(',')
                average_return.append(float(ar))
                std_return.append(float(sr))

    average_return = np.asarray(average_return)
    std_return = np.asarray(std_return)
    iterations = np.arange(len(average_return))
    ax[0].plot(iterations, average_return, 'k-', color='darkblue')
    ax[0].fill_between(iterations, average_return-std_return,
        average_return+std_return)
    ax[0].set_xticks([2,4,6,8,10])
    ax[0].set_xlabel('iteration')
    ax[0].set_ylabel('average returns')

    plt.savefig('plots_part3.png')

if __name__ == "__main__":
    plots_part1()
    plots_part3()
