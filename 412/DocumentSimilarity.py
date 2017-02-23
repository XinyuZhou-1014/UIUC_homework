from collections import defaultdict
from math import log
import numpy as np
from sklearn.decomposition import PCA
import sys
import os

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
        pca = PCA(n_components=n_components, svd_solver = 'full')
        A = self.vecDoc
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
        print(len(self.wordList))
        for i in range(5):
            l = []
            for j in res[i]:
                l.append(str(j[1]+1))
            s = ' '.join(l)
            print(s)
    
    # standard PCA test
    '''
    def pca_np(self):
        if self.vecDoc is None:
            self.vecDocGenerator()
        x = self.vecDoc
        x -= np.mean(x, axis = 0)  
        cov = np.cov(x, rowvar = False)
        evals , evecs = np.linalg.eig(cov)
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:,idx]
        evals = evals[idx]
        a = np.dot(x, evecs) 
        #print(evals)
        a = np.array(list(map(lambda x: x[:2], a)))
        self.pcaVecDoc = a.real
        evals = np.array(evals) / sum(evals)
        for i in evals:
            print(i)
    '''


if __name__ == "__main__":
    #path =  'corpus.txt'
    path = os.path.abspath(sys.argv[1])
    with open(path, 'r') as f:
        data = f.read().split('\n')
    t = DocumentSimilarty(data)
    t.show()

