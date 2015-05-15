# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:26:05 2015

@author: sxd7257
"""
import SymbolData
import Features
import numpy as NP
import matplotlib.pyplot as plt
import train
import pickle
#from PIL import Image, ImageDraw

def trainExprs(exprs, filename):
    file = open(filename,'wb')
    store = []
#    k = 0
    for expr in exprs:
        if(len(expr.symbols)==60):
            I = Features.getImgExpr(expr)
            store.append([expr, I, len(expr.symbols)])
    pickle.dump(store,file,pickle.HIGHEST_PROTOCOL)
        
def scc(I1,I2):
    X = NP.reshape(I1,(I1.size))
    Y = NP.reshape(I2,(I2.size))
    X_ctr = X-NP.mean(X)
    Y_ctr = Y-NP.mean(Y)
    varX = NP.sum(X_ctr.T*X_ctr)
    varY = NP.sum(Y_ctr.T*Y_ctr)
    covXY = NP.sum(X_ctr.T*Y_ctr)
    sqcc = covXY**2/(varX*varY)
    return sqcc
    
#exprs , classes= SymbolData.unpickleSymbols("test.dat")
#symbols = SymbolData.allSymbols(exprs)
#scale = 29
#symbols = SymbolData.normalize(symbols,scale)
#
##i=0
##for symbol in symbols:
##    print(i)
##    I = Features.features(symbol)
##    i+=1
#
##7989,12287,12288,23126,23127 test.dat
## 2467,3121,22071,22072,22731,46263 train.dat
#
## Without vertical repositioning
##6432,6433
#i=0
#for symbol in symbols[0:1000]:
#    print(i)
#    I = Features.symbolFeatures(symbol)
#    i+=1
#train.main(["-rf","RF20_FullDepthTest.mdl","trainSample","trainLgSample"])

#exprs = SymbolData.readInkmlDirectory('test','testLg',True,True)
#scale = 199
#exprs = SymbolData.normalizeExprs(exprs,scale)
#symbols = SymbolData.allSymbols(exprs)
#scale = 29
#symbols = SymbolData.normalize(symbols,scale)
#for expr in exprs[0:]:
#    Features.showImgExpr(expr)
#    plt.figure()
#    expr.plot()
#trainExprs(exprs,'testImgs\TestImgs60.mdl')
file = open('trainImgs\TrainImgs2.mdl','rb')
trainDatas = pickle.load(file)

file = open('testImgs\TestImgs2.mdl','rb')
testDatas = pickle.load(file)

#testExprs = SymbolData.readInkmlDirectory('inkml_test','lg_test')
#scale = 199
#testExprs = SymbolData.normalizeExprs(testExprs,scale)
#testExpr = testExprs[10]
testData = testDatas[10]

matchExprns = []
for trainData in trainDatas:
    if(trainData[2]==testData[2]):
        matchExprns.append(trainData)

#for trainData in trainDatas:
score = NP.zeros((len(matchExprns)))
k = 0
#testImg = Features.getImgExpr(testExpr)
for exprList in matchExprns:
#    score[k] = scc(testImg,exprList[1])
    score[k] = scc(testData[1],exprList[1])
    k+=1
ind = NP.argsort(score).astype(int)
score = score[ind]
score = score/score[-1]
matchExprSort = []
for i in ind:
    matchExprSort.append(matchExprns[i])

plt.figure()
I = NP.flipud(testData[1])
plt.imshow(I)
plt.gray()
plt.show()

for i in [-1,-2,-3,-4,-5]:
    print(score[i])
    print(matchExprSort[i][0].relations)
    plt.figure()
    I = NP.flipud(matchExprSort[i][1])
    plt.imshow(I)
    plt.gray()
    plt.show()
