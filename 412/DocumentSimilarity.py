from collections import defaultdict
from math import log
import numpy as np
from matplotlib.mlab import PCA as matPCA
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
import pickle
import json

class DocumentSimilarty():
    def __init__(self, data):
        self.data = data
        self.numDoc = len(self.data)
        self.wordDict = defaultdict(dict)
        for D in range(len(self.data)):
            doc = self.data[D].split(' ')
            for word in doc:
                if D in self.wordDict[word]:
                    self.wordDict[word][D] += 1
                else:
                    self.wordDict[word][D] = 1
        self.wordList = list(self.wordDict.keys())
        self.vecDoc = None
        self.pcaVecDoc = None

    def tf(self, t, D):
        if D >= self.numDoc:
            print('tf(%s, %s): illegal query %s.'%(t, D, D))
            return
        if t not in self.wordDict:
            print('tf(%s, %s): illegal query %s.'%(t, D, t))
            return
        
        if D not in self.wordDict[t]:
            return 0
        return self.wordDict[t][D] / len(self.data[D])

    def idf(self, t):
        return log(self.numDoc / len(self.wordDict[t]))

    def vecDocGenerator(self):
        idfDict = {}
        for word in self.wordList:
            idfDict[word] = self.idf(word)
        self.vecDoc = []
        for D in range(self.numDoc):
            vecD = [self.tf(word, D) * idfDict[word] for word in self.wordList]
            self.vecDoc.append(vecD)

    def _norm(self, v):
        s = list(map(lambda x: x * x, v))
        return sum(s) ** (1/2)

    def _dotProduct(self, v1, v2):
        res = 0
        for i in range(len(v1)):
            res += v1[i] * v2[i]
        return res

    def _pca(self, n_components=2):
        if self.vecDoc is None:
            self.vecDocGenerator()
        pca = PCA(n_components=n_components)
        A = self.vecDoc
        # without normalization
        #A = StandardScaler().fit_transform(A)
        self.pcaVecDoc = pca.fit_transform(np.array(A))

    def _ManhanttanDistance(self, v1, v2):
        res = 0
        for i in range(len(v1)):
            res += abs(v1[i] - v2[i])
        return res

    def _EuclideanDistance(self, v1, v2):
        res = 0
        for i in range(len(v1)):
            res += (v1[i] - v2[i]) ** 2
        return res ** (1/2)

    def _SupremumDistance(self, v1, v2):
        res = 0
        for i in range(len(v1)):
            res = max(res, abs(v1[i] - v2[i]))
        return res

    def _CosineSimilarity(self, v1, v2):
        res = 0
        n_v1 = self._norm(v1)
        n_v2 = self._norm(v2)
        dotProduct = self._dotProduct(v1, v2)
        return dotProduct / (n_v1 * n_v2)

    def _EucDisAfterPCA(self, v1, v2):
        if self.pcaVecDoc is None:
            print("didn't do PCA yet" , ...)
            return
        return self._EuclideanDistance(v1, v2)

    def solution(self, question):
        Q = self.numDoc - 1
        funcList = [self._ManhanttanDistance, self._EuclideanDistance, self._SupremumDistance, 
                    self._CosineSimilarity, self._EucDisAfterPCA]
        func = funcList[question]

        res = []
        if func == self._EucDisAfterPCA:
            for D in range(self.numDoc):
                res.append((func(self.pcaVecDoc[D], self.pcaVecDoc[Q]), D))
        else:
            for D in range(self.numDoc):
                res.append((func(self.vecDoc[D], self.vecDoc[Q]), D))

        res.sort(key=lambda x: x[0])
        if func == self._CosineSimilarity:
            a = res[-5:]
            a.reverse()
            return a
        return res[:5]

    def test(self):
        if self.vecDoc is None:
            self.vecDocGenerator()
        if self.pcaVecDoc is None:
            self._pca()
        d = {}
        for i in range(5):
            d[i] = self.solution(i)
        return d

    def show(self):
        res = self.test()
        for i in range(5):
            l = []
            for j in res[i]:
                l.append(str(j[1]+1))
            s = ' '.join(l)
            print(s)

    def plot(self):
        if self.vecDoc is None:
            self.vecDocGenerator()
        if self.pcaVecDoc is None:
            self._pca()

        x = list(map(lambda x: x[0], self.pcaVecDoc))
        y = list(map(lambda x: x[1], self.pcaVecDoc))
        plt.scatter(x, y)
        plt.show()
    
    def pca_np(self):
        if self.vecDoc is None:
            self.vecDocGenerator()
        print('start calc pca')
        B = self.vecDoc
        #x = [x[:100] for x in B[:100]]
        x = B
        x -= np.mean(x, axis = 0)  
        cov = np.cov(x, rowvar = False)
        evals , evecs = np.linalg.eig(cov)
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:,idx]
        evals = evals[idx]
        a = np.dot(x, evecs) 
        #resultDict = {'ev': str(ev), 'eig': str(eig), 'a': str(a)}
        #s = json.dumps(resultDict)
        #pickleSeries = str(s).encode('utf-8')
        #with open('pca_a_pickle', 'wb') as f:
        #    pickle.dump(pickleSeries, f)
        m = 0
        for i in range(len(a)):
            for j in range(len(a[i])):
                if a[i][j].imag !=0:
                    print(m, i, j, a[i][j], ...)
                    m += 1



if __name__ == "__main__":
    path =  'HW1-data/HW1-data/corpus.txt'
    #path = sys.argv[1]
    with open(path, 'r') as f:
        data = f.read().split('\n')
    t = DocumentSimilarty(data)
    #for i in [500, 207, 203, 43, 93, 430, 15, 136, 1, 4, 6, 375, 260, 3, 454]:
    #    doc = data[i-1]
    #    print('%s:' %i)
    #    print(doc)
    #    print()
    '''
    
    t._pca()
    print(t.pcaVecDoc[:5])
    t._pca()
    print(t.pcaVecDoc[:5])
    t._pca()
    print(t.pcaVecDoc[:5])
    '''
    #t.pca_np()
    d = t.test()




#A = np.matrix(t.vecDoc)
'''
A = [[    160,     2,     3,    13],
     [5,    11,    10,     8],
     [9,     7,     6,    12],
     [4,    14,    15,     1]]
pca = PCA(n_components=2)
A_std = StandardScaler().fit_transform(A)
res = pca.fit_transform(np.array(A_std))
print(res, pca.explained_variance_ratio_)
res = matPCA(np.array(A))
print(res.Y, res.fracs)
'''