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
import copy
import math
#from PIL import Image, ImageDraw

def trainExprs(exprs, filename):
    file = open(filename,'wb')
    store = []
#    k = 0
    for expr in exprs:
        if(len(expr.symbols)==7):
            I = Features.getImgExpr(expr)
            store.append([expr, I, len(expr.symbols)])
    pickle.dump(store,file,pickle.HIGHEST_PROTOCOL)
        
def scc(I1,I2):
#    I1 = NP.rint(I1/I1.max()*255).astype('uint8')
#    I2 = NP.rint(I2/I2.max()*255).astype('uint8')
    X = NP.reshape(I1,(I1.size))
    Y = NP.reshape(I2,(I2.size))
    X_ctr = X-NP.mean(X)
    Y_ctr = Y-NP.mean(Y)
    varX = NP.sum(X_ctr.T*X_ctr)
    varY = NP.sum(Y_ctr.T*Y_ctr)
    covXY = NP.sum(X_ctr.T*Y_ctr)
    sqcc = covXY**2/(varX*varY)
    return sqcc
    
#def MI(I1,I2):
##    I1 = NP.rint(I1/I1.max()*255).astype('uint8')
##    I2 = NP.rint(I2/I2.max()*255).astype('uint8')
#    mat12 = NP.zeros((256,256))
#    for i in range(I1.shape[0]):
#        for j in range(I1.shape[1]):
#            mat12[I1[i,j],I2[i,j]] += 1
#    mat12 = mat12/NP.sum(mat12)
#    I1_marg = NP.sum(mat12,axis=1)
#    I2_marg = NP.sum(mat12,axis=0)
#    H1 = -NP.sum(NP.multiply(I1_marg , NP.log2(I1_marg + (I1_marg==0))))
#    H2 = -NP.sum(NP.multiply(I2_marg , NP.log2(I2_marg + (I2_marg==0))))
#    mat12 = NP.reshape(mat12,(mat12.size))
#    H12 = -NP.sum(NP.multiply(mat12, NP.log2(mat12 + (mat12==0))))
#    mi = H1+H2-H12    
#    return(mi)

def MI(I1,I2):
#    I1 = NP.rint(I1/I1.max()*255).astype('uint8')
#    I2 = NP.rint(I2/I2.max()*255).astype('uint8')
    I1 = NP.reshape(I1,[I1.size])
    I2 = NP.reshape(I2,[I2.size])
    h = NP.histogram2d(I1,I2,2)
    mat12 = h[0]
#    mat12 = NP.zeros((2,2))
#    for i in range(I1.shape[0]):
#        for j in range(I1.shape[1]):
#            mat12[I1[i,j],I2[i,j]] += 1
    mat12 = mat12/NP.sum(mat12)
    I1_marg = NP.sum(mat12,axis=1)
    I2_marg = NP.sum(mat12,axis=0)
    H1 = -NP.sum(NP.multiply(I1_marg , NP.log2(I1_marg + (I1_marg==0))))
    H2 = -NP.sum(NP.multiply(I2_marg , NP.log2(I2_marg + (I2_marg==0))))
    mat12 = NP.reshape(mat12,(mat12.size))
    H12 = -NP.sum(NP.multiply(mat12, NP.log2(mat12 + (mat12==0))))
    mi = H1+H2-H12    
    return(mi)
    
def getSpatialFeatures(s1,s2):
    feat = NP.array([])

    x1min = s1.xmin()
    x1max = s1.xmax()
    y1min = s1.ymin()
    y1max = s1.ymax()
    x1ctr = (x1min+x1max)/2
    y1ctr = (y1min+y1max)/2
    
    x2min = s2.xmin()
    x2max = s2.xmax()
    y2min = s2.ymin()
    y2max = s2.ymax()
    x2ctr = (x2min+x2max)/2
    y2ctr = (y2min+y2max)/2
    
    xminPair = min(x1min,x2min)
    xmaxPair = max(x1max,x2max)
    yminPair = min(y1min,y2min)
    ymaxPair = max(y1max,y2max)
    diffxPair = xmaxPair-xminPair
    diffyPair = ymaxPair-yminPair
    
    diffLeft = x2min-x1min
    diffRight = x2max-x1max
    diffBottom = y2max-y1max
    diffTop = y2min-y1min
    diffHorz = x2min-x1max
    diffVert = y2min-y1max
    diffxCtr = x2ctr-x1ctr
    diffyCtr = y2ctr-y1ctr
        
    feat = NP.append(feat, [diffLeft, diffRight, diffBottom, diffTop, diffHorz, diffVert, diffxCtr, diffyCtr])
    feat = feat/diffyPair
    
    fracArea1 = ((x1max-x1min)*(y1max-y1min))/(diffxPair*diffyPair)
    fracArea2 = ((x2max-x2min)*(y2max-y2min))/(diffxPair*diffyPair)
#    areaRatio = ((x1max-x1min)*(y1max-y1min))/((x2max-x2min)*(y2max-y2min))
    Theta = math.atan2(diffyCtr,diffxCtr)
    feat = NP.append(feat,[Theta, fracArea1, fracArea2])
    
    return feat
    
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
#ascender = ['b','d', 'f', 'k', 'l', 'h', 'i', 't', '\\lambda']
#descender = ['y','p', 'q', 'j', 'g']
#centered = [')', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Y', '\\Delta', '\\alpha', '\\beta', '\\cos', '\\gamma', '\\lim', '\\log', '\\mu', '\\phi', '\\pi', '\\prime', '\\sigma', '\\sin', '\\sqrt', '\\sum', '\\tan', '\\theta', '\\}', ']', 'a',  'c', 'e', 'm', 'n', 'o', 'r', 's', 'u', 'v', 'w', 'x', 'z', '|']
#nonScripted = ['!', '(', '=', '+', '-', '.', '/','COMMA', '[', '\\div', '\\exists', '\\forall', '\\geq', '\\gt', '\\in', '\\infty', '\\int', '\\ldots', '\\leq', '\\lt', '\\neq', '\\pm', '\\rightarrow', '\\times',  '\\{']
##relations = ['Right', 'Above', 'Below', 'Sup', 'Sub', 'Inside']
#relDict = {'Right': 0, 'R': 0, 'Above': 1, 'A': 0, 'Below': 2, 'B': 2, 'Sup': 3, 'Sub': 4, 'Inside': 5, 'I': 5}
#

exprs = SymbolData.readInkmlDirectory('train','trainLg',True,True)
scale = 299
exprs = SymbolData.normalizeExprs(exprs,scale)

#trainRelFeat = []
#relClass = []
#for expr in exprs:
#    for rel in expr.relations:
#        ind = rel.find(',')
#        rel = rel[ind+2:]
#        ind = rel.find(',')
#        s1_ident = rel[0:ind]
#        rel = rel[ind+2:]
#        ind = rel.find(',')
#        s2_ident = rel[0:ind]
#        rel= rel[ind+2:]
#        ind = rel.find(',')
#        s1s2 = rel[0:ind]
#        for symbol in expr.symbols:
#            if(symbol.ident==s1_ident):
#                s1 = copy.deepcopy(symbol)
#            if(symbol.ident==s2_ident):
#                s2 = copy.deepcopy(symbol)
#        feat = getSpatialFeatures(s1,s2)
#        trainRelFeat.append(feat)
#        relClass.append(relDict[s1s2])

#symbols = SymbolData.allSymbols(exprs)
#scale = 29
#symbols = SymbolData.normalize(symbols,scale)
#for expr in exprs[0:]:
#    Features.showImgExpr(expr)
#    plt.figure()
#    expr.plot()
#trainExprs(exprs,'testImgs\TestImgs60.mdl')
######################################################################
#file = open('trainImgs\TrainImgs12.mdl','rb')
#trainDatas = pickle.load(file)
#
#file = open('testImgs\TestImgs12.mdl','rb')
#testDatas = pickle.load(file)


file = open('trainImgs7.mdl','rb')
trainDatas = pickle.load(file)

file = open('testImgs7.mdl','rb')
testDatas = pickle.load(file)

#testExprs = SymbolData.readInkmlDirectory('inkml_test','lg_test')
#scale = 199
#testExprs = SymbolData.normalizeExprs(testExprs,scale)
#testExpr = testExprs[10]
testData = testDatas[1]

matchExprns = []
for trainData in trainDatas:
    if(trainData[2]==testData[2]):
        matchExprns.append(trainData)

#for trainData in trainDatas:
scoreSCC = NP.zeros((len(matchExprns)))
scoreMI = NP.zeros((len(matchExprns)))
k = 0
#testImg = Features.getImgExpr(testExpr)
import time
t = time.time()
times = []
for exprList in matchExprns:
    scoreSCC[k] = scc(testData[1],exprList[1])
    scoreMI[k] = MI(testData[1],exprList[1])
    k+=1
print(time.time() - t)
indSCC = NP.argsort(scoreSCC).astype(int)
scoreSCC = scoreSCC[indSCC]
scoreSCC = scoreSCC/scoreSCC[-1]

#indMI = NP.argsort(scoreMI).astype(int)
#scoreMI = scoreMI[indMI]
#scoreMI = scoreMI/scoreMI[-1]

matchExprSortSCC = []
for i in indSCC:
    matchExprSortSCC.append(matchExprns[i])

#matchExprSortMI = []
#for i in indMI:
#    matchExprSortMI.append(matchExprns[i])

print(testData[0].relations)
plt.figure()
I = NP.flipud(testData[1])
plt.imshow(I)
plt.gray()
plt.show()
testData[0].plot()

for i in [-1,-2,-3]:
    print(scoreSCC[i])
    print(matchExprSortSCC[i][0].relations)
    plt.figure()
    I = NP.flipud(matchExprSortSCC[i][1])
    plt.imshow(I)
    plt.gray()
    plt.show()
    matchExprSortSCC[i][0].plot()

#    print(scoreMI[i])
#    print(matchExprSortMI[i][0].relations)
#    plt.figure()
#    I = NP.flipud(matchExprSortMI[i][1])
#    plt.imshow(I)
#    plt.gray()
#    plt.show()
#    matchExprSortMI[i][0].plot()
    
######################################################################
