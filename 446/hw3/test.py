import pandas
from LearningMethods import LearningMethods
from LearningWithStop import LearningWithStop
from gen import gen
from numpy import dot, sign
from matplotlib import pyplot as plt
import logging
logging.basicConfig(level=logging.DEBUG)

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


class Trainer():
    def __init__(self, **kw):
        self.gen = gen
        self.set_param(**kw)
        
    def set_param(self, **kw):
        for key, val in kw.items():
            if key in ['l', 'm', 'n', 'number_of_instances', 'noise', 'usage']:
                self.__setattr__(key, val)

    def _data_generator(self):
        self.data_description = 'l = %s, m = %s, n = %s, number_of_instances = %s, noise = %s. ' % \
        (self.l, self.m, self.n, self.number_of_instances, self.noise)
                            
        logging.info('Generating data with parameters:')
        logging.info(self.data_description)
        
        if self.usage is None:
            self.usage = 'Train'
            logging.info('As usage set to default: Train')
        else:
            logging.info('As usage: %s'%usage)
        self.y, self.x = self.gen(self.l, self.m, self.n, self.number_of_instances, self.noise)
        self.data_description += 'Size of x: %s, Size of y: %s' % (self.x.shape, self.y.shape)
        return {'y': self.y, 'x': self.x, 'description': self.data_description, 'usage': self.usage}

    def data_generator_by_self_params(self, usage=None):
        return self._data_generator(l=self.l, n=self.n, m=self.m, 
            number_of_instances=self.number_of_instances, noise=self.noise)
    def data_generator_by_dict(self, param_dict):
        usage = param_dict.get('usage', None)
        return self._data_generator(l=param_dict['l'], n=param_dict['n'], m=param_dict['m'], 
            number_of_instances=param_dict['number_of_instances'], noise=param_dict['noise'], usage=usage)
    
    def data_generator(self, **kw):
        self.usage = None
        self.set_param(**kw)
        dct = self._data_generator()
        numTuningData = kw['number_of_instances'] // 10
        self.D1_x = dct['x'][:numTuningData]
        self.D1_y = dct['y'][:numTuningData]
        self.D2_x = dct['x'][numTuningData: 2 * numTuningData]
        self.D2_y = dct['y'][numTuningData: 2 * numTuningData]
        #self.D2_x = dct['x'][:numTuningData]
        #self.D2_y = dct['y'][:numTuningData]
        dct['D1_x'] = self.D1_x
        dct['D1_y'] = self.D1_y
        dct['D2_x'] = self.D2_x
        dct['D2_y'] = self.D2_y
        logging.debug('size of D1_x: %s, size of D2_x: %s'%(self.D1_x.shape, self.D2_x.shape))
        return dct


    def learning(self, learning_algorithm, x, y, initDict=None, times=1):
        logging.info('Method: {0}, Data Info {1}, times: {2}'.format(
            learning_algorithm, (self.l, self.m, self.n), times))
        self.resDict = LearningMethods.learning(learning_algorithm, x, y, initDict, times)

    def learningWithStop(self, learning_algorithm, x, y, initDict=None, times=1):
        logging.info('Learn with stopping criteria: 1000 continuous example without mistake.')
        logging.info('Method: {0}, Data Info {1}, times: {2}'.format(
            learning_algorithm, (self.l, self.m, self.n), times))
        self.resDict = LearningWithStop.learning(learning_algorithm, x, y, initDict, times)

    def _unpack_resDict(self, learning_rate, margin):
        self.w, self.theta, self.mistake, self.mistake_list = self.resDict[(learning_rate, margin)]

    def test(self, x, learning_rate, margin):
        self._unpack_resDict(learning_rate, margin)
        return sign(dot(x, self.w) + self.theta)

    def error_estimate(self, x, y, learning_rate, margin):
        yp = self.test(x, learning_rate, margin)
        err = 0
        for i in range(len(y)):
            if y[i] != yp[i]:
                err += 1
        return err / len(y)

    def mistakeCount(self, learning_rate, margin):
        self._unpack_resDict(learning_rate, margin)
        return self.mistake





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


def problem_2tune():
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


def problem_2plot():
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

problem_2plot()