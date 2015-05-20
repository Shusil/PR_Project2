import numpy as NP
import sklearn
import sklearn.ensemble
import sklearn.decomposition
import Features
import SymbolData
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from functools import reduce
import pickle
import copy

#1-NN classifier. Pretends to be a sklearn model, so we can use the same code.
class OneNN:
    def __init__(self):
        self.classes = NP.array([])
        self.features = NP.array([])

    def fit(self, features, classes):
        self.features = features
        self.classes = classes
        return self

    def predict(self, samples):
        return NP.array(list( map ((lambda s: self.nearest(s)), samples))) #have no idea why it breaks if I go straight to array.

    def nearest(self, sample):
        self.sqrs = NP.power(self.features - sample, 2)
        self.sums = NP.sum(self.sqrs, axis=1)
        self.dists = NP.sqrt(self.sums)
        self.idx = NP.argmin(self.dists)
        return self.classes[self.idx]

def makeRF():
    #play with the options once we have a reasonable set of features to experiment with.
    return sklearn.ensemble.RandomForestClassifier(n_estimators=20, n_jobs = -1, verbose=1)

def makeET():
    #play with the options once we have a reasonable set of features to experiment with.
    return sklearn.ensemble.ExtraTreesClassifier()
        
def train(model, training, keys, pca_num=None):
    if model == "1nn":
        model = OneNN()
    elif model == "rf":
        model = makeRF()
    training = SymbolData.normalize(training, 29)
    f = Features.features(training)
    pca = None
    if (pca_num != None):
        pca = sklearn.decomposition.PCA(n_components=pca_num)
        pca.fit(f)
        f = pca.transform(f)
    model.fit(Features.features(training), SymbolData.classNumbers(training, keys))
    return (model, pca)


def classifyExpressions(expressions, keys, model, pca, out, renormalize=True, showAcc = False):
    #this sort of does double duty, since it is both classifying the symbols
    # with side effects and returnning stuff to evaluate the results.
    # Bad style. Sorry.

    cors = list([])
    preds = list([])
    tot = len(expressions)
    i = 0
    
#    s = showAcc
    
    for expr in expressions:
        print(expr)
        correct, predicted =  classifyExpression(expr, keys, model, pca, renormalize)
        #assert (len(correct) == len(predicted))
#        if s:
#            cors = cors + [correct]
#        if correct == None:
#            s = False
#            cors = [[], []]
        
        preds = preds + [predicted]
        print(i, "/",tot)
        f = (lambda p: keys[p])
            #    expr.classes = map (f, preds[i])

        expr.writeLG(out ,clss =  map (f, preds[i]) )
        i+=1

        #print (correct, " -> ", predicted)        
#    if s :
#        print (cors)
#        print( "Accuracy on testing set : ", accuracy_score(NP.concatenate(cors), NP.concatenate(preds)))
    return (cors, preds)

with open('RF20_FullDepthBoxFeat.mdl', 'rb') as f:
    model, pca, keys =  pickle.load(f)
#import SymbolData
#file = open('trainImgs1.mdl','rb')
#trainDatas = pickle.load(file)
trainDatas = None
def setTrainData(d):
    global trainDatas
    trainDatas = d
    
cache = {}
def classifySymbol(symb, keys=keys, model=model, pca=pca, renormalize=True):
#    orig = copy.deepcopy(symb) 
    global cache
#    print(cache)
    if str(symb) in cache:
#        print('cache hit')
        return cache[str(symb)]
    symbl = [copy.deepcopy(symb)]
    if renormalize:
        symbl = SymbolData.normalize(symbl, 29)
    f = Features.features(symbl)
    
    if (pca != None):
        f = pca.transform(f)
    pred = model.predict_proba(f)
    predChar = model.predict(f)
    predChar = keys[predChar]
    cache[str(symb)] = [pred, predChar]
    return pred, predChar
    
    
def classifyExpression(expression, keys, model, pca, renormalize=True):
    symbs = expression.symbols
    if renormalize:
        symbs = SymbolData.normalize(symbs, 29)
    f = Features.features(symbs)
    if (len (symbs) == 0):
        print(expression.name, " has no valid symbols!")
        return ([], [])
    if (pca != None):
        f = pca.transform(f)
    pred = model.predict(f)
    assert (max(pred) < len(keys))
    f = (lambda p: keys[p])
    expression.classes = map (f, pred)
    return (NP.array(SymbolData.classNumbers(symbs, keys)), pred)


def getMatchingExpression(testExpr):
    matchExprns = []
    for trainData in trainDatas:
        if(trainData[2]==len(testExpr.symbols)):
            matchExprns.append(trainData)
    
    #for trainData in trainDatas:
    scoreSCC = NP.zeros((len(matchExprns)))
#    scoreMI = NP.zeros((len(matchExprns)))
    k = 0
    testImg = Features.getImgExpr(testExpr)
    for exprList in matchExprns:
        scoreSCC[k] = scc(testImg,exprList[1])
    #    scoreMI[k] = MI(testData[1],exprList[1])
        k+=1
    indSCC = NP.argsort(scoreSCC).astype(int)
    scoreSCC = scoreSCC[indSCC]
    scoreSCC = scoreSCC/scoreSCC[-1]
    
#    indMI = NP.argsort(scoreMI).astype(int)
#    scoreMI = scoreMI[indMI]
#    scoreMI = scoreMI/scoreMI[-1]
    
    matchExprSortSCC = []
    for i in indSCC:
        matchExprSortSCC.append(matchExprns[i])
    
    return(matchExprSortSCC[-1])
#    return(matchExprns[indSCC[-1]])

def scc(I1,I2):
    I1 = NP.rint(I1/I1.max()*255).astype('uint8')
    I2 = NP.rint(I2/I2.max()*255).astype('uint8')
    X = NP.reshape(I1,(I1.size))
    Y = NP.reshape(I2,(I2.size))
    X_ctr = X-NP.mean(X)
    Y_ctr = Y-NP.mean(Y)
    varX = NP.sum(X_ctr.T*X_ctr)
    varY = NP.sum(Y_ctr.T*Y_ctr)
    covXY = NP.sum(X_ctr.T*Y_ctr)
    sqcc = covXY**2/(varX*varY)
    return sqcc
    
def MI(I1,I2):
    I1 = NP.rint(I1/I1.max()*255).astype('uint8')
    I2 = NP.rint(I2/I2.max()*255).astype('uint8')
    mat12 = NP.zeros((256,256))
    for i in range(I1.shape[0]):
        for j in range(I1.shape[1]):
            mat12[I1[i,j],I2[i,j]] += 1
    mat12 = mat12/NP.sum(mat12)
    I1_marg = NP.sum(mat12,axis=1)
    I2_marg = NP.sum(mat12,axis=0)
    H1 = -NP.sum(NP.multiply(I1_marg , NP.log2(I1_marg + (I1_marg==0))))
    H2 = -NP.sum(NP.multiply(I2_marg , NP.log2(I2_marg + (I2_marg==0))))
    mat12 = NP.reshape(mat12,(mat12.size))
    H12 = -NP.sum(NP.multiply(mat12, NP.log2(mat12 + (mat12==0))))
    mi = H1+H2-H12    
    return(mi)

