'''
Contains LearningWithStop and LearningHingeLoss
'''

from math import sqrt
import numpy as np
from numpy import array, dot
from LearningMethods import LearningMethods
import logging
from matplotlib import pyplot as plt
from random import randrange


class LearningWithStop(LearningMethods):
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
        no_mistake_length = 0
        mistake_list = []
        w = array(w).astype('float')
        while no_mistake_length < 1000:
            i = randrange(x.shape[0])
            xi, yi = x[i], y[i]
            predict = yi * (dot(w, xi) + theta)
            if predict <= 0:
                mistake += 1
                no_mistake_length = 0
            else:
                no_mistake_length += 1
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
        no_mistake_length = 0
        w = array(w).astype('float')
        while no_mistake_length < 1000:
            i = randrange(x.shape[0])
            xi, yi = x[i], y[i]
            predict = yi * (dot(w, xi) + theta)
            if predict <= 0:
                mistake += 1
                no_mistake_length = 0
            else:
                no_mistake_length += 1
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
        no_mistake_length = 0
        w = array(w).astype('float')
        while no_mistake_length < 1000:
            i = randrange(x.shape[0])
            xi, yi = x[i], y[i]
            predict = yi * (dot(w, xi) + theta)
            if predict <= 0:
                mistake += 1
                no_mistake_length = 0
            else:
                no_mistake_length += 1
            if predict > 1:
                pass
            else:
                g = - yi * (np.append(xi, 1))
                G = sum(np.apply_along_axis(lambda x: x ** 2, axis=0, arr=g))
                w -= learning_rate * g[:-1] / sqrt(G)
                theta -= learning_rate * g[-1] / sqrt(G)
            mistake_list.append(mistake)
        return w, theta, mistake, mistake_list



class LearningHingeLoss():
