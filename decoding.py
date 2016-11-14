# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 21:48:39 2016

@author: yuhui
"""

import scipy.io as sio
import numpy as np
import cPickle as pickle
from hmmlearn.hmm import MultinomialHMM
import os
import matplotlib.pyplot as plt
from pybrain.structure import RecurrentNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError

def load_dataset():
    test_cell_file = 'cellts_test.mat'
    train_cell_file = 'cellts_train.mat'
    test_pos_file = 'POS_test.mat'
    train_pos_file = 'POS_train.mat'
    
    test_cell = sio.loadmat(test_cell_file)['cell_test'] #0*n array
    train_cell = sio.loadmat(train_cell_file)['cell_train']
    test_post, test_posx = sio.loadmat(test_pos_file)['post_test'], sio.loadmat(test_pos_file)['posx_test'] #array
    train_post, train_posx = sio.loadmat(train_pos_file)['post_train'], sio.loadmat(train_pos_file)['posx_train']
    
    return test_cell, train_cell, test_post, test_posx, train_post, train_posx

def cell_preprocess(train_cell, train_post):
    cell_number = train_cell.shape[1]
    cell_matrix = np.zeros((train_post.shape[0],cell_number))
    for i in range(cell_number):
        compar = train_cell[0][i]
        for j in range(len(compar)):
            index = np.argmin(np.abs(train_post - compar[j]))
            cell_matrix[index][i] = 1
    
    return cell_matrix

def pos_preprocess(train_posx, window_size):
    map_size = np.ceil((train_posx.max() - train_posx.min())) / window_size
    posx = np.ceil((train_posx - train_posx.min()) / window_size)
    tmp = 0
    posx_frequent = []
    for i in range(posx.shape[1]):
        tmp += posx[:,i]
    for unit in set(list(tmp)):
        posx_frequent.append(list(tmp).count(unit))
    transit = np.zeros((len(set(tmp)),len(set(tmp))))
    for i in range(tmp.shape[0]):
        if i != 1:
            transit[list(set(tmp)).index(tmp[i-1]), list(set(tmp)).index(tmp[i])] += 1
    posx_frequent = np.array(posx_frequent, dtype=float)
    posx_frequent /= (train_posx.shape[0] *1.0)
    return map_size, posx, posx_frequent, tmp, transit

def transit_matrix(cell_matrix):
    tmp = 0
    for i in range(cell_matrix.shape[1]):
        tmp += np.power(10,i) * cell_matrix[:,i]
    transit = np.zeros((len(set(tmp)),len(set(tmp))))
    stat_frequent = []
    for unit in set(tmp):
        stat_frequent.append(list(tmp).count(unit))
    stat_frequent = np.array(stat_frequent, dtype=float)
    stat_frequent /= tmp.shape[0]
    for i in range(tmp.shape[0]):
        if i != 1:
            transit[list(set(tmp)).index(tmp[i-1]), list(set(tmp)).index(tmp[i])] += 1
    for i in range(len(set(tmp))):
        transit[i] /= (np.sum(transit[i]) *1.0)
    
    return stat_frequent, transit, tmp

def get_likelihood(posx, cell_matrix_set, posx_set):
    likelihood_matrix = np.zeros((len(set(posx_set)),len(set(cell_matrix_set)))) #[posx_count, cell_matrix_count]
    for i in range(posx.shape[0]):
        likelihood_matrix[list(set(posx_set)).index(posx[i]),
                          list(set(cell_matrix_set)).index(cell_matrix_set[i])] += 1
    for i in range(likelihood_matrix.shape[0]):
        likelihood_matrix[i] /= (np.sum(likelihood_matrix[i]) * 1.0)
    
    return likelihood_matrix

def bayesian_function(test_spike, cell_matrix_set, posx_set, pri_posx, pri_spike, likelihood_matrix):
    tmp = 0
    result = []
    for i in range(test_spike.shape[1]):
        tmp += np.power(10,i) * test_spike[:,i]
    for i in range(len(tmp)):
        cell_index = list(set(cell_matrix_set)).index(tmp[i])
        likelihood = likelihood_matrix[:,cell_index]
        pri_cell = pri_spike[cell_index]
        prob = np.multiply(pri_posx,likelihood) * 1.0 / pri_cell
        result.append(list(set(posx_set))[np.argmax(prob)])
    
    return np.array(result,dtype=float)

def Gaussian_kernel(x, Bin_Size):
    mean = np.mean(x)
    return_array = np.exp(-np.power((x-mean),2)/2/Bin_Size/Bin_Size)
    return return_array

def rnn_model(X, Xtest, y, yTest):
    n = RecurrentNetwork()
    n.addInputModule(LinearLayer(X.shape[1], name='in'))
    n.addModule(SigmoidLayer(X.shape[1], name='hidden'))
    n.addOutputModule(LinearLayer(1, name='out'))
    n.addConnection(FullConnection(n['in'], n['hidden'], name='c1'))
    n.addConnection(FullConnection(n['hidden'], n['out'], name='c2'))
    n.addRecurrentConnection(FullConnection(n['hidden'], n['hidden'], name='c3'))
    n.sortModules() #build network
    DStrain=SupervisedDataSet(X.shape[1],y.shape[1])
    for i in range(len(y)):
        DStrain.addSample(X[i],y[i]) #add train samples
    DStest = SupervisedDataSet(Xtest.shape[1],yTest.shape[1])
    for i in range(len(yTest)):
        DStest.addSample(Xtest[i],yTest[i]) #add test samples
    print 'start training rnn'
    trainer = BackpropTrainer(n, DStrain, verbose = True, learningrate=0.01)
    trainer.trainUntilConvergence(maxEpochs=100)
    trnresult = percentError(trainer.testOnClassData(), DStrain['target'])
    testResult = percentError(trainer.testOnClassData(dataset=DStest), DStest['target'])
    print("epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % testResult)
    prediction=[]
    for ipr,target in DStest:
        prediction.append(n.activate(ipr))
    with open('rnn_state.pkl','wb') as f:
        pickle.dump(prediction, f)
    plt.figure()
    plt.plot(np.arange(0,len(prediction)), prediction, 'ro--', label='predict number')
    plt.plot(np.arange(0,len(yTest)), yTest, 'ko-', label='true number')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig('rnn_result.jpg')
    plt.close()
    error = model_evaluate(np.array(prediction), yTest)
    return error
    
    
def model_evaluate(prediction, yTest):
    right = (prediction - yTest) * 1.0 / (yTest.max()-yTest.min())
    return np.sum(np.abs(right)) / len(yTest)

def data_save(Bin_size):
    test_cell, train_cell, test_post, test_posx, train_post, train_posx = load_dataset()
    train_matrix = cell_preprocess(train_cell, train_post)
    test_matrix = cell_preprocess(test_cell, test_post)
    map_size_train, pr_train_posx, pri_train_posx, train_posx_set, train_posx_transit = pos_preprocess(train_posx, Bin_size)
    print 'there are %d classes' %map_size_train
    map_size_test, pr_test_posx, pri_test_posx, test_posx_set, test_posx_transit = pos_preprocess(test_posx, Bin_size)
    print 'check classes number, which is', map_size_test
    
    pri_cell_train, transit_train, train_cell_set = transit_matrix(train_matrix)
    pri_cell_test, transit_test, test_cell_set = transit_matrix(test_matrix)
    print 'transit matrix built'
    with open('train.pkl','wb') as f:
        pickle.dump(train_matrix, f)
        pickle.dump(pr_train_posx, f)
        pickle.dump(train_cell_set, f)
        pickle.dump(train_posx_set, f)
        pickle.dump(pri_train_posx, f)
        pickle.dump(pri_cell_train, f)
        pickle.dump(train_posx_transit, f)
    with open('test.pkl','wb') as f:
        pickle.dump(test_matrix, f)
        pickle.dump(pr_test_posx, f)
        pickle.dump(test_cell_set, f)
        pickle.dump(test_posx_set, f)
        pickle.dump(pri_test_posx, f)
        pickle.dump(pri_cell_test, f)
        pickle.dump(test_posx_transit, f)
    likelihood_matrix = get_likelihood(pr_train_posx, train_cell_set, train_posx_set)
    print 'likelihood done'
    with open('likelihood.pkl','wb') as f:
        pickle.dump(likelihood_matrix,f)

def HMM(transmat, likelihood, train_matrix, test_matrix):
    model = MultinomialHMM(n_components=transmat.shape[0])
    model.emissionprob_ = likelihood
    model.transmat_ = transmat
    print 'HMM start training'
    model.fit(train_matrix)
    log_prob, state_sequence = model.decode(test_matrix)
    print 'HMM training done'
    return log_prob, state_sequence

def HMM_input_map(matrix_set):
    HMM_dic = {}
    for unit in set(matrix_set):
        HMM_dic[unit] = list(set(matrix_set)).index(unit)
    HMM_matrix = np.atleast_2d(np.zeros((matrix_set.shape[0],1)))
    for i in range(matrix_set.shape[0]):
        HMM_matrix[i,0] = HMM_dic[matrix_set[i]]
    return HMM_matrix
    
def main():
    Bin_size = 20
    if not os.path.exists('likelihood.pkl'):    
        data_save(Bin_size)
    with open('train.pkl','rb') as f:
        train_matrix = pickle.load(f)
        pr_train_posx = pickle.load(f)
        train_cell_set = pickle.load(f)
        train_posx_set = pickle.load(f)
        pri_train_posx = pickle.load(f)
        pri_cell_train = pickle.load(f)
        train_posx_transit = pickle.load(f)
    with open('test.pkl','rb') as f:
        test_matrix = pickle.load(f)
        pr_test_posx = pickle.load(f)
        test_cell_set = pickle.load(f)
        test_posx_set = pickle.load(f)
        pri_test_posx = pickle.load(f)
        pri_cell_test = pickle.load(f)
        test_posx_transit = pickle.load(f)
    with open('likelihood.pkl','rb') as f:
        likelihood_matrix = pickle.load(f)
    #Naive Bayesian Model
    result = bayesian_function(test_matrix, train_cell_set, train_posx_set, pri_train_posx, pri_cell_train, likelihood_matrix)
    print result
    plt.figure()
    plt.plot(np.arange(0,len(pr_test_posx)), result, 'ro--', label='predict number')
    plt.plot(np.arange(0,len(pr_test_posx)), pr_test_posx, 'ko-', label='true number')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig('bayesian_result.jpg')
    plt.close()
    error_bayesian = model_evaluate(result, test_posx_set)
    print 'bayesian model error: ', error_bayesian
    with open('Bayesian_state.pkl','wb') as f:
        pickle.dump(result, f)
    #RNN Model
    error_rnn = rnn_model(train_matrix, test_matrix, pr_train_posx, pr_test_posx)
    print 'rnn model error: ', error_rnn
    #HMM Model
    train_matrix = np.array(HMM_input_map(train_cell_set),dtype=int)
    test_matrix = np.array(HMM_input_map(test_cell_set),dtype=int)
    log_prob, state_sequence = HMM(train_posx_transit, likelihood_matrix, train_matrix, test_matrix)
    plt.figure()
    plt.plot(np.arange(0,len(pr_test_posx)), state_sequence, 'ro--', label='predict number')
    plt.plot(np.arange(0,len(pr_test_posx)), pr_test_posx, 'ko-', label='true number')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig('HMM_result.jpg')
    plt.close()
    with open('HMM_state.pkl','wb') as f:
        pickle.dump(state_sequence, f)
        pickle.dump(log_prob,f)
    error_hmm = model_evaluate(state_sequence, test_posx_set)
    print 'HMM model error: ', error_hmm
    with open('error.txt','w') as f:
        f.write(str(error_bayesian))
        f.write('\n')
        f.write(str(error_hmm))
        f.write('\n')
        f.write(str(error_rnn))

if __name__ == '__main__':
    main()