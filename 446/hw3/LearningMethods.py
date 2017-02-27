from math import sqrt
import numpy as np
from numpy import array, dot
import logging
#logging.basicConfig(level=logging.INFO)

class LearningMethods():
    
    @staticmethod
    def _perceptron(x, y, w, theta, learning_rate, margin=0):
        if margin < 0:
            raise ValueError('Negative margin <%s> is prohibited in perceptron.'%margin)
        if learning_rate <= 0:
            raise ValueError('Negative learning rate <%s> is prohibited in perceptron.'%learning_rate)

        if margin == 0:
            if learning_rate != 1:
                logging.debug('No margin. Set learning rate to 1.')
                learning_rate = 1
            logging.debug('Perception. Tune: None.')
        else:
            if margin != 1:
                logging.debug('Margin of Perceptron is fixed at 1.')
                margin = 1
            logging.debug('Perceptron with margin = 1. Tune: learning rate (%s).'%learning_rate)
        
        mistake = 0
        mistake_list = []
        w = array(w).astype('float')
        for i in range(y.shape[0]):
            xi, yi = x[i], y[i]
            predict = yi * (dot(w, xi) + theta)
            if predict <= 0:
                mistake += 1
            if predict > margin:
                pass
            else:
                w += learning_rate * yi * xi
                theta += learning_rate * yi 
            mistake_list.append(mistake)
        return w, theta, mistake, mistake_list
    
    @staticmethod
    def _winnow(x, y, w, theta, learning_rate, margin=0):
        if margin < 0:
            raise ValueError('Negative margin <%s> is prohibited in winnow.'%margin)
        if learning_rate <= 1:
            raise ValueError('Learning rate less than 1 <%s> is prohibited in winnow.'%learning_rate)
        if margin == 0:
            logging.debug('Winnow. Tune: learning rate (%s).'%learning_rate)
        else:
            logging.debug('Winnow with margin. Tune: margin (%s) and learning rate (%s).' %
                (margin, learning_rate))
        logging.debug('Theta is fixed to -n.')
        theta = - x.shape[1]

        mistake = 0
        mistake_list = []
        w = array(w).astype('float')
        for i in range(y.shape[0]):
            xi, yi = x[i], y[i]
            predict = yi * (dot(w, xi) + theta)
            if predict <= 0:
                mistake += 1
            if predict > margin:
                pass
            else:
                w *= learning_rate ** (yi * xi)
                # do not update theta 
            mistake_list.append(mistake)
        return w, theta, mistake, mistake_list

    @staticmethod
    def _AdaGrad(x, y, w, theta, learning_rate, margin=None):
        if learning_rate <= 0:
            raise ValueError('Negative learning rate <%s> is prohibited in AdaGrad.'%learning_rate)
        logging.debug('AdaGrad (no margin). Tune: Learning rate (%s)' %learning_rate)
        
        mistake = 0
        mistake_list = []
        w = array(w).astype('float')
        for i in range(y.shape[0]):
            xi, yi = x[i], y[i]
            predict = yi * (dot(w, xi) + theta)
            if predict <= 0:
                mistake += 1
            if predict > 1:
                pass
            else:
                g = - yi * (np.append(xi, 1))
                G = sum(np.apply_along_axis(lambda x: x ** 2, axis=0, arr=g))
                w -= learning_rate * g[:-1] / sqrt(G)
                theta -= learning_rate * g[-1] / sqrt(G)
            mistake_list.append(mistake)
        return w, theta, mistake, mistake_list


    @classmethod
    def _learning_deploy(cls, method, x, y, w_init, theta_init, learning_rate, margin, times=1):
        funcDict = {'Perceptron': cls._perceptron, 'Winnow': cls._winnow, 
                    'AdaGrad': cls._AdaGrad}
        try:
            func = funcDict[method]
        except KeyError as e:
            raise(e)

        if times <= 0:
            logging.warning('Iterate 0 times?')
            return 
        w, theta = w_init, theta_init
        mistake_total = 0
        for i in range(times):
            w, theta, mistake, mistake_list = func(x, y, w, theta, learning_rate, margin)
            mistake_total += mistake
        return w, theta, mistake_total, mistake_list

    @classmethod
    def learning(cls, learning_algorithm, x, y, initDict=None, times=1):
        if callable(initDict):
            n = x.shape[1]
            initDict = initDict(n)
        dct = initDict[learning_algorithm]
        method, w, theta = dct['method'], dct['w'], dct['theta']
        learning_rate_list, margin_list = dct['learning rate'], dct['margin']

        
        res = {}
        for learning_rate in learning_rate_list:
            for margin in margin_list:
                res[(learning_rate, margin)] = cls._learning_deploy(
                    method, x, y, w, theta, learning_rate, margin, times)
        return res

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)