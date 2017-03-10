from LearningMethods import LearningMethods
from LearningMethodsAdditional import LearningWithStop, LearningHingeLoss
import logging
from gen import gen
from numpy import dot, sign

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

    def learningHingeLoss(self, learning_algorithm, x, y, initDict=None, times=1):
        logging.info('Learning Mistake and HingeLoss.')
        logging.info('Method: {0}, Data Info {1}, times: {2}'.format(
            learning_algorithm, (self.l, self.m, self.n), times))
        self.resDict = LearningHingeLoss.learning(learning_algorithm, x, y, initDict, times)

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

    def hinge_and_mis(self, xt, yt, w, theta):
        yp = dot(xt, w) + theta
        hinge = 0
        mis = 0
        for i in range(len(yt)):
            hinge += max(0, 1-yt[i] * yp[i])
            mis += abs(sign(yp[i]) - yt[i]) // 2
        return hinge, mis
