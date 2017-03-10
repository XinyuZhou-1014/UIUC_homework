from LearningMethods import LearningMethods
from LearningMethodsAdditional import LearningWithStop
from Trainer import Trainer
#from gen import gen
import numpy as np
from numpy import dot, sign
from matplotlib import pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)

def initDictGenerator(n):
    initDict = {
        'Perceptron':
        {
            'method': 'Perceptron', 
            'w': [0] * n, 
            'theta': 0, 
            'learning rate': [1], 
            'margin': [0]
        },
        'Perceptron with margin':
        {
            'method': 'Perceptron', 
            'w': [0] * n, 
            'theta': 0, 
            'learning rate': [1.5, 0.25, 0.03, 0.005, 0.001], 
            'margin': [1]
        },
        'Winnow':
        {
            'method': 'Winnow', 
            'w': [1] * n, 
            'theta': -n, 
            'learning rate': [1.1, 1.01, 1.005, 1.0005, 1.0001], 
            'margin': [0]
        },
        'Winnow with margin':
        {
            'method': 'Winnow', 
            'w': [1] * n, 
            'theta': -n, 
            'learning rate': [1.1, 1.01, 1.005, 1.0005, 1.0001], 
            'margin': [2.0, 0.3, 0.04, 0.006, 0.001]
        },
        'AdaGrad':
        {
            'method': 'AdaGrad', 
            'w': [0] * n, 
            'theta': 0, 
            'learning rate': [1.5, 0.25, 0.03, 0.005, 0.001], 
            'margin': [None] # Useless
        }
    }
    return initDict


def problem_1b(n=500):
    # change n for two size of datasets
    algorithmList = ['Perceptron', 'Perceptron with margin', 'Winnow', 'Winnow with margin', 'AdaGrad']

    t = Trainer()
    d = t.data_generator(l=10, m=100, n=n, number_of_instances=50000, noise=False)

    x, y = d['x'], d['y'] 
    D1_x, D1_y, D2_x, D2_y = d['D1_x'], d['D1_y'], d['D2_x'], d['D2_y']
    initDict = initDictGenerator(n=t.n)
    for algorithm in algorithmList: 
        algorithmInit = initDict[algorithm]
        learningRateList = algorithmInit['learning rate']
        marginList = algorithmInit['margin']
        t.learning(algorithm, D1_x, D1_y, initDict=initDict, times=20)
        for lr in learningRateList:
            for mg in marginList:
                err_rate = t.error_estimate(t.D2_x, t.D2_y, lr, mg)
                mistake = t.mistakeCount(lr, mg)
                print('LR: {0: >6s}, MG: {1: >6s}, ER: {2: >6s}, Mis: {3: >6s}'.format(
                    str(lr), str(mg), str(err_rate), str(mistake)))


def problem_1c(n=500):
    algorithmList = ['Perceptron', 'Perceptron with margin', 'Winnow', 'Winnow with margin', 'AdaGrad']
    bestParaList = {500:  [(1, 0), (0.03, 1), (1.1, 0), (1.1, 2.0), (0.25, 1)], \
                    1000: [(1, 0), (0.03, 1), (1.1, 0), (1.1, 2.0), (0.25, 1)]}
    t = Trainer()
    d = t.data_generator(l=10, m=100, n=n, number_of_instances=50000, noise=False)
    x, y = d['x'], d['y'] 
    initDict = initDictGenerator(n=t.n)
    color = 'rgbyk'
    for idx in range(len(algorithmList)): 
        algorithm = algorithmList[idx]
        algorithmInit = initDict[algorithm]
        if n in bestParaList:
            lr, mg = bestParaList[n][idx]
        else:
            lr, mg = bestParaList[500][idx]
        algorithmInit['learning rate'] = [lr]
        algorithmInit['margin'] = [mg]
        t.learning(algorithm, x, y, initDict={algorithm: algorithmInit}, times=1)
        t._unpack_resDict(lr, mg)
        plt.plot(t.mistake_list, color[idx])
    plt.legend(algorithmList, loc='best')
    plt.title('Plots for n = %s'%n)
    plt.show()


def problem_2_tuning():
    # change n for two size of datasets
    algorithmList = ['Perceptron', 'Perceptron with margin', 'Winnow', 'Winnow with margin', 'AdaGrad']

    for n in range(40, 240, 40):
        print()
        t = Trainer()
        d = t.data_generator(l=10, m=20, n=n, number_of_instances=50000, noise=False)

        x, y = d['x'], d['y'] 
        D1_x, D1_y, D2_x, D2_y = d['D1_x'], d['D1_y'], d['D2_x'], d['D2_y']
        initDict = initDictGenerator(n=t.n)
        for algorithm in algorithmList: 
            algorithmInit = initDict[algorithm]
            learningRateList = algorithmInit['learning rate']
            marginList = algorithmInit['margin']
            t.learning(algorithm, D1_x, D1_y, initDict=initDict, times=20)
            for lr in learningRateList:
                for mg in marginList:
                    err_rate = t.error_estimate(t.D2_x, t.D2_y, lr, mg)
                    mistake = t.mistakeCount(lr, mg)
                    print('LR: {0: >6s}, MG: {1: >6s}, ER: {2: >6s}, Mis: {3: >6s}'.format(
                        str(lr), str(mg), str(err_rate), str(mistake)))

'''
#what a beautiful graph, but I misunderstood the question
def problem_2_plot():
    algorithmList = ['Perceptron', 'Perceptron with margin', 'Winnow', 'Winnow with margin', 'AdaGrad']
    bestParaList = {40:  [(1, 0), (0.25, 1), (1.1, 0), (1.1, 2.0), ( 1.5, 1)],
                    80:  [(1, 0), (0.03, 1), (1.1, 0), (1.1, 2.0), (0.25, 1)], 
                    120: [(1, 0), (0.03, 1), (1.1, 0), (1.1, 2.0), (0.25, 1)], 
                    160: [(1, 0), (0.03, 1), (1.1, 0), (1.1, 2.0), (0.25, 1)], 
                    200: [(1, 0), (0.25, 1), (1.1, 0), (1.1, 2.0), ( 1.5, 1)]}

    for n in range(40, 240, 40):
        print()
        plt.figure()
        t = Trainer()
        d = t.data_generator(l=10, m=20, n=n, number_of_instances=50000, noise=False)
        x, y = d['x'], d['y'] 
        initDict = initDictGenerator(n=t.n)
        color = 'rgbyk'
        for idx in range(len(algorithmList)): 
            algorithm = algorithmList[idx]
            algorithmInit = initDict[algorithm]
            if n in bestParaList:
                lr, mg = bestParaList[n][idx]
            else:
                lr, mg = bestParaList[500][idx]
            algorithmInit['learning rate'] = [lr]
            algorithmInit['margin'] = [mg]
            t.learningWithStop(algorithm, x, y, initDict={algorithm: algorithmInit}, times=1)
            print(len(t.resDict[lr, mg][3]))
            t._unpack_resDict(lr, mg)
            plt.plot(t.mistake_list, color[idx])
        plt.legend(algorithmList, loc='best')
        plt.title('Plots for n = %s'%n)
        plt.savefig('problem2plot_%s.png'%n, dpi=144)
    plt.show()
'''

def problem_2_plot():
    algorithmList = ['Perceptron', 'Perceptron with margin', 'Winnow', 'Winnow with margin', 'AdaGrad']
    bestParaList = {40:  [(1, 0), (0.25, 1), (1.1, 0), (1.1, 2.0), ( 1.5, 1)],
                    80:  [(1, 0), (0.03, 1), (1.1, 0), (1.1, 2.0), (0.25, 1)], 
                    120: [(1, 0), (0.03, 1), (1.1, 0), (1.1, 2.0), (0.25, 1)], 
                    160: [(1, 0), (0.03, 1), (1.1, 0), (1.1, 2.0), (0.25, 1)], 
                    200: [(1, 0), (0.25, 1), (1.1, 0), (1.1, 2.0), ( 1.5, 1)]}

    record = {}
    for algorithm in algorithmList:
        record[algorithm] = []
    for n in range(40, 240, 40):
        print()
        t = Trainer()
        d = t.data_generator(l=10, m=20, n=n, number_of_instances=50000, noise=False)
        x, y = d['x'], d['y'] 
        initDict = initDictGenerator(n=t.n)
        color = 'rgbyk'
        
        for idx in range(len(algorithmList)): 
            algorithm = algorithmList[idx]
            
            algorithmInit = initDict[algorithm]
            if n in bestParaList:
                lr, mg = bestParaList[n][idx]
            else:
                lr, mg = bestParaList[500][idx]
            algorithmInit['learning rate'] = [lr]
            algorithmInit['margin'] = [mg]
            t.learningWithStop(algorithm, x, y, initDict={algorithm: algorithmInit}, times=1)
            print(len(t.resDict[lr, mg][3]))
            t._unpack_resDict(lr, mg)
            record[algorithm].append(t.mistake_list[-1])

    i = 0
    for algorithm in algorithmList:
        plt.plot(range(40, 240, 40), record[algorithm], color[i])
        i += 1
    plt.legend(algorithmList, loc='best')
    plt.xlabel('Number of total features')
    plt.xticks(range(40, 240, 40))
    plt.ylabel('Total Mistakes before stop')
    #plt.title('Plots for n = %s'%n)
    plt.savefig('p2plot.png', dpi=144)
    plt.show()


def problem_3_dataGenerator():
    l = 10
    n = 1000
    for m in [100, 500, 1000]:
        t = Trainer()
        train = t.data_generator(l=l, m=m, n=n, number_of_instances=50000, noise=True)
        test = t.data_generator(l=l, m=m, n=n, number_of_instances=10000, noise=False)
        np.save('p3trainX_m=%s'%m, train['x'])
        np.save('p3trainY_m=%s'%m, train['y'])
        np.save('p3testX_m=%s'%m, test['x'])
        np.save('p3testY_m=%s'%m, test['y'])

def problem_3_pureDataGenerator():
    l = 10
    n = 1000
    for m in [100, 500, 1000]:
        t = Trainer()
        train = t.data_generator(l=l, m=m, n=n, number_of_instances=50000, noise=False)
        np.save('p3pureX_m=%s'%m, train['x'])
        np.save('p3pureY_m=%s'%m, train['y'])


def problem_3_dataLoader(m, data='train'):
    dct = {}
    data = data.lower()
    if data not in ['train', 'test', 'pure']:
        raise("Invalid data parameter: <%s>."%data)
    if data == 'pure':
        logging.info('Using pure data as train')
    dct['x'] = np.load('p3%sX_m=%s.npy'%(data, m))
    dct['y'] = np.load('p3%sY_m=%s.npy'%(data, m))
    numTuningData = dct['x'].shape[0] // 10
    dct['D1_x'] = dct['x'][:numTuningData]
    dct['D1_y'] = dct['y'][:numTuningData]
    dct['D2_x'] = dct['x'][numTuningData: 2 * numTuningData]
    dct['D2_y'] = dct['y'][numTuningData: 2 * numTuningData]
    return dct

def problem_3_tuning():
    algorithmList = ['Perceptron', 'Perceptron with margin', 'Winnow', 'Winnow with margin', 'AdaGrad']
    for m in [100, 500, 1000]:
        print()

        d = problem_3_dataLoader(m, 'train')
        x, y = d['x'], d['y'] 
        D1_x, D1_y, D2_x, D2_y = d['D1_x'], d['D1_y'], d['D2_x'], d['D2_y']
        t = Trainer()
        t.set_param(l=10, m=m, n=x.shape[1], number_of_instances=x.shape[0])
        initDict = initDictGenerator(n=t.n)
        for algorithm in algorithmList: 
            algorithmInit = initDict[algorithm]
            learningRateList = algorithmInit['learning rate']
            marginList = algorithmInit['margin']
            t.learning(algorithm, D1_x, D1_y, initDict=initDict, times=20)
            for lr in learningRateList:
                for mg in marginList:
                    err_rate = t.error_estimate(D2_x, D2_y, lr, mg)
                    mistake = t.mistakeCount(lr, mg)
                    print('LR: {0: >6s}, MG: {1: >6s}, ER: {2: >6s}, Mis: {3: >6s}'.format(
                        str(lr), str(mg), str(err_rate), str(mistake)))


def problem_3_train_and_evaluate():
    algorithmList = ['Perceptron', 'Perceptron with margin', 'Winnow', 'Winnow with margin', 'AdaGrad']
    bestParaList = {100:  [(1, 0), (0.005, 1), (1.1, 0), (1.1, 0.3), ( 1.5, 1)],
                    500:  [(1, 0), (  1.5, 1), (1.1, 0), (1.1, 0.3), ( 1.5, 1)], 
                    1000: [(1, 0), ( 0.25, 1), (1.1, 0), (1.1, 0.3), (0.25, 1)]} 

    for m in [100, 500, 1000]:
        print()
        #plt.figure()
        d = problem_3_dataLoader(m)
        x, y = d['x'], d['y']
        d = problem_3_dataLoader(m, 'test')
        xtest, ytest = d['x'], d['y']  
        t = Trainer()
        t.set_param(l=10, m=m, n=x.shape[1], number_of_instances=x.shape[0])
        initDict = initDictGenerator(n=t.n)
        #color = 'rgb'
        for idx in range(len(algorithmList)): 
            algorithm = algorithmList[idx]
            algorithmInit = initDict[algorithm]
            if m in bestParaList:
                lr, mg = bestParaList[m][idx]
            else:
                lr, mg = bestParaList[500][idx]
            algorithmInit['learning rate'] = [lr]
            algorithmInit['margin'] = [mg]
            t.learning(algorithm, x, y, initDict={algorithm: algorithmInit}, times=20)
            err_rate = t.error_estimate(xtest, ytest, lr, mg)
            print('LR: {0: >6s}, MG: {1: >6s}, ER: {2: >6s}'.format(
                str(lr), str(mg), str(err_rate)))

'''
def problem_3_test():
    algorithmList = ['Perceptron', 'Perceptron with margin', 'Winnow', 'Winnow with margin', 'AdaGrad']
    bestParaList = {100:  [(1, 0), (0.001, 1), (1.1, 0), (1.1,  0.04), ( 1.5, 1)],
                    500:  [(1, 0), ( 0.03, 1), (1.1, 0), (1.1,   2.0), ( 1.5, 1)], 
                    1000: [(1, 0), ( 0.03, 1), (1.1, 0), (1.1, 0.006), (0.25, 1)]} 

    for m in [100, 500, 1000]:
        print()
        #plt.figure()
        d = problem_3_dataLoader(m, 'train')
        x, y = d['x'], d['y']
        d = problem_3_dataLoader(m, 'test')
        xtest, ytest = d['x'], d['y']  
        t = Trainer()
        t.set_param(l=10, m=m, n=x.shape[1], number_of_instances=x.shape[0])
        initDict = initDictGenerator(n=t.n)
        color = 'rgbyk'
        for idx in range(len(algorithmList)): 
            algorithm = algorithmList[idx]
            algorithmInit = initDict[algorithm]
            if m in bestParaList:
                lr, mg = bestParaList[m][idx]
            else:
                lr, mg = bestParaList[500][idx]
            algorithmInit['learning rate'] = [lr]
            algorithmInit['margin'] = [mg]
            for i in range(1, 21, 4):
                t.learning(algorithm, x, y, initDict={algorithm: algorithmInit}, times=i)
                err_rate = t.error_estimate(xtest, ytest, lr, mg)
                print(i)
                print('LR: {0: >6s}, MG: {1: >6s}, ER: {2: >6s}'.format(
                    str(lr), str(mg), str(err_rate)))

'''

def problem_4():
    i = 0
    t = Trainer()
    l, m, n = 10, 20, 40
    l1 = []
    l2 = []
    color = 'rgbymc'
    lr, mg = (0.25, 1)
    initDict = initDictGenerator(n=n)
    algorithm = 'AdaGrad'
    algorithmInit = initDict[algorithm]
    algorithmInit['learning rate'] = [lr]
    algorithmInit['margin'] = [mg]
    for j in range(50):
        d = t.data_generator(l=l, m=m, n=n, number_of_instances=10000, noise=True)
        x, y = d['x'], d['y']
        t.learningHingeLoss(algorithm, x, y, initDict={algorithm: algorithmInit}, times=1)
        res = t.resDict[(lr, mg)]
        w, theta = res[0], res[1]
        hinge, mis = t.hinge_and_mis(x, y, w, theta)
        l1.append(hinge)
        l2.append(mis)
    plt.plot(range(1, 51), l1, color[i])
    plt.plot(range(1, 51), l2, color[i+1])
    plt.xlabel('Datasets')
    plt.ylabel('Total value of Hinge Loss / Misclassification Loss')
    plt.yticks(range(0, 3000, 300))
    plt.legend(['Hinge Loss', 'Misclassification Loss'])
    plt.savefig('p4plot.png', dpi=144)
    plt.show()
