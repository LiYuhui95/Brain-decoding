# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 16:37:09 2016

@author: yuhui
"""
import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np

def plot_compar(state_file, yTest):
    with open(state_file,'rb') as f:
        prediction = pickle.load(f)
        prediction = np.array(prediction)
    print prediction.size
    print prediction
    plt.figure()
    plt.plot(np.arange(4500,4500+len(prediction[4500:5500])), prediction[4500:5500], 'ro--', label='predict position')
    plt.plot(np.arange(4500,4500+len(yTest[4500:5500])), yTest[4500:5500], 'ko-', label='true position')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title('Recurrent Neural Network')
    plt.savefig('rnn.jpg')
    plt.close()

def main():
    test_file = 'test.pkl'
    state_file = 'rnn_state.pkl'
    with open(test_file,'rb') as f:
        _ = pickle.load(f)
        _ = pickle.load(f)
        _ = pickle.load(f)
        test_posx = pickle.load(f)
    plot_compar(state_file, test_posx)

if __name__ == '__main__':
    main()