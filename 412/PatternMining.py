from collections import defaultdict, Counter, OrderedDict 


def read(): # for hackerrank
    data = []
    s = input()
    while s is not None:
        data.append(s)
        try:
            s = input()
        except:
            break
    return data


def parseData(raw_data):
    data = raw_data.split('\n')
    min_sup = int(data[0])
    data = data[1:]
    data = list(map(lambda x: set(x.split(' ')), data))
    return min_sup, data


def getItems(data, min_sup):
    # return all len-1 values of feature and their sup (larger than min_sup)
    counter = Counter()
    for ex in data:
        counter.update(ex)
    # delete item with sup less than min_sup
    toPop = []
    for key, val in counter.items():
        if val < min_sup:
            toPop.append(key)
    for key in toPop:
        counter.pop(key)
    
    # sort candidate features by sup
    items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    return items


def scanPattern(data, patternList):
    res = [0] * len(patternList)
    for ex in data:
        for i in range(len(patternList)):
            pattern = patternList[i]
            if pattern.issubset(ex):
                res[i] += 1
    return res


def output(res):
    res = list(map(lambda x: [x[1], sorted(list(x[0]))], res))
    res.sort(key=lambda x: (-x[0], x[1]))
    string = ''
    for i in res:
        if len(i[1]) == 0:
            continue
        s = '[' + ' '.join(i[1]) + ']'
        string += '%s %s\n' %(i[0], s)
    return string


class PatternMinerNode():
    def __init__(self, head, tail):
        self.head = head
        self.tail = tail
        self.children = []
        self.parent = None

    def addChild(self, child):
        self.children.append(child)
        child.parent = self


def allPatterns(data, min_sup):
    def aprioriTreeGenerator(root, min_sup, res):
        # head, tail: list, pattern: set
        if len(root.tail) == 0:
            return 
        head, tail = root.head, root.tail
        #print('<head and tail>:', head, tail)
        
        patternList = []
        for feature in tail:
            patternList.append(set(head + [feature]))
        #print('<pattern list> :', patternList)
        supCounter = scanPattern(data, patternList)

        for i in range(len(supCounter)):
            if supCounter[i] >= min_sup:
                res.append([patternList[i], supCounter[i]])
                newNode = PatternMinerNode(head + [tail[i]], tail[i+1:])
                root.addChild(newNode)
                aprioriTreeGenerator(newNode, min_sup, res)

    items = getItems(data, min_sup)
    featureList = list(map(lambda x: x[0], items))
    root = PatternMinerNode([], featureList)
    res = []
    aprioriTreeGenerator(root, min_sup, res)
    return output(res)
    

def closePatterns(data, min_sup):
    def addClosePattern(res, pattern, sup):
        flag = True
        toPop = []
        for i in range(len(res)):
            existPattern, existSup = res[i]
            if pattern.issubset(existPattern) and sup <= existSup:
                flag = False
            if existPattern.issubset(pattern) and existSup <= sup:
                toPop.append(i)
        toPop.reverse()
        for i in toPop:
            del res[i]
        if flag:
            res.append([pattern, sup])

    def closePatternTreeGenerator(root, min_sup, res):
        # head, tail: list, pattern: set
        if len(root.tail) == 0:
            return 
        head, tail = root.head, root.tail
        #print('<head and tail>:', head, tail)
        
        patternList = []
        for feature in tail:
            patternList.append(set(head + [feature]))
        #print('<pattern list> :', patternList)
        supCounter = scanPattern(data, patternList)

        for i in range(len(supCounter)):
            if supCounter[i] >= min_sup:
                addClosePattern(res, patternList[i], supCounter[i])
                newNode = PatternMinerNode(head + [tail[i]], tail[i+1:])
                root.addChild(newNode)
                closePatternTreeGenerator(newNode, min_sup, res)

    def finalBattle(res):
        res.sort(key=lambda x:-x[1])
        toPop = set()
        for i in range(len(res)):
            p0, s0 = res[i]
            for j in range(i+1, len(res)):
                p1, s1 = res[j]
                if p1.issubset(p0):
                    toPop.add(j)
        toPop = sorted(list(toPop), reverse=True)
        for i in toPop:
            del res[i]


    items = getItems(data, min_sup)
    featureList = list(map(lambda x: x[0], items))
    root = PatternMinerNode([], featureList)
    res = []
    closePatternTreeGenerator(root, min_sup, res)

    finalBattle(res)
    return output(res)


def maxPatterns(data, min_sup, ):
    def maxTreeGenerator(root, min_sup, res):
        head, tail = root.head, root.tail
        #print('<head and tail>:', head, tail)
        
        maxPattern = set(head + tail)
        patternList = [maxPattern]
        for feature in tail:
            patternList.append(set(head + [feature]))
        #print('<pattern list> :', patternList)
        supCounter = scanPattern(data, patternList)
        #print('<sup counter>  :', supCounter)
        if supCounter[0] >= min_sup: # global prune
            #print('<max pattern>  :', maxPattern)
            for pattern, _ in res:
                if maxPattern.issubset(pattern):
                    return 
            res.append([maxPattern, supCounter[0]])
            return
        
        newTail = []
        for i in range(1, len(patternList)):
            if supCounter[i] >= min_sup:
                newTail.append(tail[i-1])

        for i in range(len(newTail)):
            newNode = PatternMinerNode(head + [newTail[i]], newTail[i+1:])
            root.addChild(newNode)
            maxTreeGenerator(newNode, min_sup, res)
        if len(newTail) == 0 and len(tail)!=0:
            supCounter = scanPattern(data, [set(root.head)])
            #print('<sup counter>  :', supCounter)
            if supCounter[0] >= min_sup:
                maxPattern = set(head)
                for pattern, _ in res:
                    if len(maxPattern - pattern) == 0:
                        return 
                res.append([maxPattern, supCounter[0]])
                return

    items = getItems(data, min_sup)
    featureList = list(map(lambda x: x[0], items))
    root = PatternMinerNode([], featureList)
    res = []
    maxTreeGenerator(root, min_sup, res)
    return output(res)


def run(raw_data):
    min_sup, data = parseData(raw_data)
    res = allPatterns(data, min_sup)
    print(res)
    res = closePatterns(data, min_sup)
    print(res)
    res = maxPatterns(data, min_sup)
    print(res)


if __name__ == '__main__':
    raw_data = '''2
    1 2 5
    2 4
    2 3
    1 2 4
    1 3
    2 3
    1 3
    1 2 3 5
    1 2 3'''
    run(raw_data)
    raw_data = '''2
    data mining
    frequent pattern mining
    mining frequent patterns from the transaction dataset
    closed and maximal pattern mining'''
    run(raw_data)
    raw_data = '''2
    B A C E D
    A C
    C B D'''
    run(raw_data)
    raw_data = '''2
    a c d e f
    a b e
    c e f
    a c d f
    c e f'''
    run(raw_data)

# FP tree generator -- don't know when to stop so cancelled
'''
class FPTreeNode():
    def __init__(self, label):
        self.label = label
        self.sup = 0
        self.children = OrderedDict()
        self.parent = None

    def addChild(self, label):
        self.sup += 1
        if label in self.children:
            return None
        else:
            node = FPTreeNode(label)
            node.parent = self
            self.children[label] = node
            return node
        
    def getChild(self, label):
        return self.children[label]

    def __str__(self):
        return str((self.label, self.sup))
    __repr__ = __str__

    def prt(root):
        def _prt(root, spaces):
            if not root:
                return
            print(spaces + str(root))
            for label, child in root.children.items():
                _prt(child, spaces+'    ')
        _prt(root, '')
def fpTree(data, min_sup):
    def fpTreeGenerator(data, root, featureList):
        # featureList: a list with the first item as label, and follwing items as all 
        # fp tree nodes with the same label, which is the backtracker of the fp tree
        for ex in data:
            curr = root
            for featureInfo in featureList:
                feature = featureInfo[0]
                if feature in ex:
                    child = curr.addChild(feature)
                    if child:
                        featureInfo.append(child)
                    curr = curr.getChild(feature)
            curr.sup += 1

    items = getItems(data, min_sup)
    featureList = list(map(lambda x: [x[0]], items))
    # generate F-P Tree
    root = FPTreeNode('root')
    root.data = data
    fpTreeGenerator(data, root, featureList)

    # try generate fp tree of i3:
    nodes = featureList[2][1:]
    newdata = []
    for node in nodes:
        ex = []
        sup = node.sup
        node = node.parent
        while node.label!= 'root':
            ex.append(node.label)
            node = node.parent
        for _ in range(sup):
            newdata.append(ex)
    newroot = FPTreeNode('root')
    newFeatureList = [['2'], ['1'], ['3']]
    fpTreeGenerator(newdata, newroot, newFeatureList)
    newroot.prt()
    # generate frequent patterns by fp tree and featureList
'''