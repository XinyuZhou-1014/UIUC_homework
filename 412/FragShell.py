import math
import copy

def read():
    data = []
    s = input()
    while s is not None:
        data.append(s)
        try:
            s = input()
        except:
            break
    return data

def preprocessing(raw_data):
    numShell = int(raw_data.pop(0))
    row = raw_data[0] # 'a1 b2 c3 d4 e5'
    row = row.split(' ')
    n = len(row)
    sizeShell = math.ceil(n / numShell)
    listSizeShell = [sizeShell] * numShell
    listSizeShell = [0] + listSizeShell
    for i in range(1, numShell+1):
        listSizeShell[i] += listSizeShell[i-1]
    categories = list(map(lambda x: x[0], row))
    groups = []
    for i in range(1, len(listSizeShell)):
        groups.append(categories[listSizeShell[i-1]: listSizeShell[i]])
    return categories, groups

def getSubsets(group):
    #['a', 'b', 'c'] -> [['a'], ['b'], ['c'], ['a', 'b'], ['a', 'c'], ['b', 'c'], ['a', 'b', 'c']]
    def _subsets_onestep(group, idx, res):
        if idx >=len(group):
            return res
        resres = []
        for i in res:
            tmp = copy.copy(i)
            tmp.append(group[idx])
            resres.append(tmp)
        res.extend(resres)
        return _subsets_onestep(group, idx+1, res)
    res = _subsets_onestep(group, 0, [[]])[1:] # remove the empty set
    res.sort(key = lambda x: (len(x), x[0]))
    return res
    
def ex2dict(s):
    d = {}
    for i in s:
        d[i[0]] = i[1]
    return d 

def pools(allEx):
    def pool(ex):
        return set(ex.split(' '))
    allPool = {}
    exPools = []
    for ex in allEx:
        s = pool(ex)
        for key, val in ex2dict(s).items():
            val = key + val
            if key not in allPool:
                allPool[key] = set([val])
            else:
                allPool[key].add(val)
        exPools.append(s)
    for key, val in allPool.items():
        allPool[key] = sorted(list(val))
    return allPool, exPools

def subset2comb(subset, allPool): #['a', 'b'] -> [['a1', 'b1'], ['a1', 'b2'], ...]
    def _one_step(subset, index, res, d):
        if index >= len(subset):
            #return list(map(lambda x: x.lstrip(' '), res))
            return res
        feature = subset[index]
        fVals = d[feature]
        tmp = []
        for r in res:
            for fVal in fVals:
                tmp.append(r+[fVal])
        return _one_step(subset, index+1, tmp, d)
    return _one_step(subset, 0, [[]], allPool)

def subsets2combs(subsets, allPool):
    res = []
    for subset in subsets:
        res.extend(subset2comb(subset, allPool))
    return res

def foo(raw_data):
    cats, grps = preprocessing(raw_data)
    res = []
    for grp in grps:
        res.append(getSubsets(grp))

    allPool, exPools = pools(raw_data)
    resres = []
    for subsets in res:
        resres.append(subsets2combs(subsets, allPool))
    return resres, exPools

def satisfy(ex, item):
    for i in item:
        if i not in ex:
            return False
    return True


#raw_data = ['3', 'a1 b2 c1 d1 e1 f1 g1 h1 i1 j1 k1 l1 m1 n1', 'a1 b2 c1 d2 e1', 'a1 b2 c1 d1 e2', 'a2 b1 c1 d1 e2', 'a2 b1 c1 d1 e3']
raw_data = ['2', 'a1 b2 c1 d1 e1', 'a1 b2 c1 d2 e1', 'a1 b2 c1 d1 e2', 'a2 b1 c1 d1 e2', 'a2 b1 c1 d1 e3']

items, exPools = foo(raw_data)
output = []
for shell in items:
    for item in shell:
        num = 0
        for ex in exPools:
            if satisfy(ex, item):
                num += 1
        if num != 0:
            output.append(' '.join(item) + ' : %d'%num)
    output.append('')

for i in output[:-1]:
    print(i)
