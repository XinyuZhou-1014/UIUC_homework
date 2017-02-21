from math import log
import numpy as np
import pandas as pd
from random import random
from pandas import DataFrame as df
from matplotlib import pyplot as plt

def cost_function(X, y, w, theta):
    res = 0
    n = len(X)
    for i in range(len(X)):
        row = X[i]
        res += (np.dot(w, row) + theta - y[i]) ** 2
    return res / n

def gradient(X, y, w, theta):
    def d_of_w(z, y_i, w, theta):
        return 2 * z * (np.dot(w, z)) + 2 * theta * z - 2 * z * y_i
    def d_of_theta(z, y_i, w, theta):
        return 2 * np.dot(w, z) + 2 * theta - 2 * y_i
    
    n = len(X)
    res = [0 for i in range(len(X[0]))]
    g_theta = 0
    g_w = np.array(res).astype('float')
    for i in range(len(X)):
        row = X[i]
        g_w += np.array(d_of_w(row, y[i], w, theta))
        g_theta += d_of_theta(row, y[i], w, theta)
    g_w /= n
    g_theta /= n
    return (g_w, g_theta)

def train(features, label, itertimes=1000, alpha=0.1, alphaDecrease=True, epsilon=0.01, silent=False):
    features = features.astype('float')
    X = np.array(features)
    yy = []
    for i in label:
        if i == '+':
            yy.append(1)
        else:
            yy.append(-1)
    y = np.array(yy).astype('float')

    w = [random() for i in range(features.shape[1])]
    w = np.array(w).astype('float')
    theta = 1

    record = []
    trueAlpha = alpha
    for i in range(itertimes):
        if alphaDecrease:
            trueAlpha = alpha / (int( i / 100) + 1)
        cost = cost_function(X, y, w, theta)
        record.append(cost)
        g_w, g_theta = gradient(X, y, w, theta)
        if sum(abs(g_w)) / len(g_w) < epsilon:
            break
        w -= trueAlpha * g_w
        theta -= trueAlpha * g_theta
        if not silent:
            if (i + 1) % 250 == 0:
                t = '%s times, alpha: %.4f, error = %s' %(i + 1, trueAlpha, cost)
                print(t)
    plt.show()
    return w, theta, record
    
def test(features, label, w, theta):
    features = features.astype('float')
    X = np.array(features)
    yy = []
    for i in label:
        if i == '+':
            yy.append(1)
        else:
            yy.append(-1)
    y = np.array(yy).astype('float')

    predict = []
    for i in range(len(X)):
        row = X[i]
        if np.dot(w, row) + theta >= 0:
            predict.append(1)
        else:
            predict.append(-1)

    n = len(y)
    res = 0
    for i in range(n):
        if y[i] != predict[i]:
            res += 1
    return res, n, predict

def rendering(raw_data):
    data = []
    attribute_list = []
    for line in raw_data:
        line_splited = line.split(' ')
        if line[0] in '01':
            data.append(line.rstrip().split(','))
        elif len(line_splited) > 1 and line_splited[0] == '@attribute':
            attribute_list.append(line.split(' ')[1])

    data = df(data, columns=attribute_list)
    return data

def choose_test_data(dataList, n):
    testData = dataList[n]
    dataList = dataList[:n] + dataList[n+1:]
    trainData = pd.concat(dataList, ignore_index=True)
    return trainData, testData

def problem_1(silent=True):
    numFold=5
    filenames = ['badges.example.fold%s.arff' %i for i in range(1, numFold+1)]
 
    dataList = []
    for filename in filenames:
        with open(filename, 'r') as file:
            raw_data = file.readlines()
            data = rendering(raw_data)
            dataList.append(data)

    w_sets = []
    theta_sets = []
    for cv_number in range(numFold):
        trainData, testData = choose_test_data(dataList, cv_number)
        
        attribute_list = testData.columns
        trainLabel = trainData['Class']
        trainFeatures = trainData[attribute_list[:-1]]
        testLabel = testData['Class']
        testFeatures = testData[attribute_list[:-1]]
        w, theta, record = train(trainFeatures, trainLabel, itertimes=10000, alpha = 0.01, 
            alphaDecrease=False, epsilon=0.001, silent=silent)
        error, total, predict = test(testFeatures, testLabel, w, theta)
        print('Accuracy: %.2f%%' %(100 - (error / total) * 100))
        w_sets.append(w)
        theta_sets.append(theta)
    return w_sets

def get_stump_features(filename):
    with open(filename, 'r') as file:
        s = file.readlines()
    train = []
    test = []
    for row in s:
        row = row.split(' ')
        if len(row) == 2:
            train.append(list(row[0]))
            test.append(list(row[1])[:-1])
    train = df(train).transpose()
    test = df(test).transpose()
    return train, test

def problem_5(silent=True):
    numFold = 5
    with open("badges.example.all.arff", 'r') as file:
        raw_data = file.readlines()

    dataList = []
    filenames = ['badges.example.fold%s.arff' %i for i in range(1, 6)]
    for filename in filenames:
        with open(filename, 'r') as file:
            raw_data = file.readlines()
            data = rendering(raw_data)
            dataList.append(data)

    Accuracy = []
    for cv_number in range(numFold):
        trainData, testData = choose_test_data(dataList, cv_number)
        attribute_list = testData.columns
        trainLabel = trainData['Class']
        testLabel = testData['Class']
        
        trainFeatures, testFeatures = get_stump_features('stumpfeatures_fold%s.txt'%(cv_number+1))
        w, theta, record = train(trainFeatures, trainLabel, itertimes=10000, alpha=0.015, 
            alphaDecrease=False, epsilon=0.001, silent=silent)
        error, total, predict = test(testFeatures, testLabel, w, theta)

        Accuracy.append(error / total)
        print('Accuracy: %.2f%%' %(100 - (error / total) * 100))

    print('Summary:', ...)
    for a in Accuracy:
        print('Accuracy: %.2f%%' %(100 - a * 100))