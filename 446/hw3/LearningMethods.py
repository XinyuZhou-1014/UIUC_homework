from math import sqrt
import numpy as np
from numpy import array, dot
import logging
logging.basicConfig(level=logging.INFO)

class LearningMethods():

    def __init__(self):
        pass

    @staticmethod
    def _perceptron(x, y, w, theta, learning_rate, margin=0):
        if margin < 0:
            raise ValueError('Negative margin <%s> is prohibited in perceptron.'%margin)
        if learning_rate <= 0:
            raise ValueError('Negative learning rate <%s> is prohibited in perceptron.'%learning_rate)

        if margin == 0:
            if learning_rate != 1:
                logging.info('No margin. Set learning rate to 1.')
                learning_rate = 1
            logging.info('Perception. Tune: None.')
        else:
            if margin != 1:
                logging.info('Margin of Perceptron is fixed at 1.')
                margin = 1
            logging.info('Perceptron with margin = 1. Tune: learning rate (%s).'%learning_rate)
        
        w = array(w).astype('float')
        for i in range(y.shape[0]):
            xi, yi = x[i], y[i]
            predict = yi * (dot(w, xi) + theta)
            if predict > margin:
                pass
            else:
                w += learning_rate * yi * xi
                theta += learning_rate * yi 
        return w, theta
    
    @staticmethod
    def _winnow(x, y, w, theta, learning_rate, margin=0):
        if margin < 0:
            raise ValueError('Negative margin <%s> is prohibited in winnow.'%margin)
        if learning_rate <= 1:
            raise ValueError('Learning rate less than 1 <%s> is prohibited in winnow.'%learning_rate)
        if margin == 0:
            logging.info('Winnow. Tune: learning rate (%s).'%learning_rate)
        else:
            logging.info('Winnow with margin. Tune: margin (%s) and learning rate (%s).' %
                (margin, learning_rate))
        logging.info('Theta is fixed to -n.')
        theta = - x.shape[1]

        w = array(w).astype('float')
        for i in range(y.shape[0]):
            xi, yi = x[i], y[i]
            predict = yi * (dot(w, xi) + theta)
            if predict > margin:
                pass
            else:
                w *= learning_rate ** (yi * xi)
                # do not update theta 
        return w, theta

    @staticmethod
    def _AdaGrad(x, y, w, theta, learning_rate, margin=0):
        if learning_rate <= 0:
            raise ValueError('Negative learning rate <%s> is prohibited in AdaGrad.'%learning_rate)
        logging.info('AdaGrad (no margin). Tune: Learning rate (%s)' %learning_rate)
        
        w = array(w).astype('float')
        for i in range(y.shape[0]):
            xi, yi = x[i], y[i]
            predict = yi * (dot(w, xi) + theta)
            if predict > 1:
                pass
            else:
                g = - yi * (np.append(xi, 1))
                G = sum(np.apply_along_axis(lambda x: x ** 2, axis=0, arr=g))
                w -= learning_rate * g[:-1] / sqrt(G)
                theta -= learning_rate * g[-1] / sqrt(G)
        return w, theta


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
        if times == 1:
            return func(x, y, w_init, theta_init, learning_rate, margin)
        else:
            w, theta = w_init, theta_init
            for i in range(times):
                w, theta = func(x, y, w, theta, learning_rate, margin)
            return w, theta

    @classmethod
    def learning(cls, learning_algorithm, x, y, times=1):
        n = x.shape[1]
        initDict = cls._initDictGenerator(n)
        dct = initDict[learning_algorithm]
        method, w, theta = dct['method'], dct['w'], dct['theta']
        learning_rate_list, margin_list = dct['learning rate'], dct['margin']

        
        res = {}
        for learning_rate in learning_rate_list:
            for margin in margin_list:
                res[(learning_rate, margin)] = \
                cls._learning_deploy(method, x, y, w, theta, learning_rate, margin, times)
        return res

    @classmethod
    def _initDictGenerator(cls, n):
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
                'margin': [0] # Useless
            }
        }
        return initDict

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)