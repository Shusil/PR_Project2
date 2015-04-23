# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:26:05 2015

@author: sxd7257
"""
import SymbolData
import Features
import numpy as NP
import matplotlib.pyplot as plt
from scipy.spatial import distance
import train
#from PIL import Image, ImageDraw

def doBboxOverlap(bbox1,bbox2):
    xctr1 = (bbox1[0]+bbox1[2])/2
    yctr1 = (bbox1[1]+bbox1[3])/2
    w1 = (bbox1[2]-bbox1[0])
    h1 = (bbox1[3]-bbox1[1])

    xctr2 = (bbox2[0]+bbox2[2])/2
    yctr2 = (bbox2[1]+bbox2[3])/2
    w2 = (bbox2[2]-bbox2[0])
    h2 = (bbox2[3]-bbox2[1])
    
    return((abs(xctr1-xctr2)*2<=(w1+w2))&(abs(yctr1-yctr2)*2<=(h1+h2)))

def doBboxContain(bbox1,bbox2):
    contain = True      #doesnot contain
    xmin = min(bbox1[0],bbox2[0])
    ymin = min(bbox1[1],bbox2[1])
    xmax = max(bbox1[2],bbox2[2])
    ymax = max(bbox1[3],bbox2[3])
    if(((xmax-xmin)==(bbox1[2]-bbox1[0]))&((ymax-ymin)==(bbox1[3]-bbox1[1]))):
        contain = False         #contain
    if(((xmax-xmin)==(bbox2[2]-bbox2[0]))&((ymax-ymin)==(bbox2[3]-bbox2[1]))):
        contain = False         #contain
    return(contain)

def getBboxMeanDist(bbox1,bbox2):
    xctr1 = (bbox1[0]+bbox1[2])/2
    yctr1 = (bbox1[1]+bbox1[3])/2

    xctr2 = (bbox2[0]+bbox2[2])/2
    yctr2 = (bbox2[1]+bbox2[3])/2

    x = xctr1-xctr2
    y = yctr1-yctr2
    dist = NP.linalg.norm(NP.array([x,y]))
    return dist    

def getMeanDist(stk1,stk2):
    pts1 = NP.asarray(stk1.asPoints())
    meanPts1 = NP.mean(pts1, axis=0)
    pts2 = NP.asarray(stk2.asPoints())
    meanPts2 = NP.mean(pts2, axis=0)
    meanVec = meanPts1-meanPts2
    meanDist = NP.linalg.norm(meanVec)
    return(meanDist)

def getPerimeter(stk):
    w = stk.xmax()-stk.xmin()
    h = stk.ymax()-stk.ymin()
    return(2*(w+h))

def getMinDist(stk1,stk2):
    pts1 = NP.asarray(stk1.asPoints())
    pts2 = NP.asarray(stk2.asPoints())
    dist = distance.cdist(pts1,pts2)
    minD = NP.min(dist)
    z = NP.asarray(NP.where(dist==minD))
    indP1 = z[0][0]
    indP2 = z[1][0]
    return(minD,indP1,indP2)
    
def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def dist(p1,p2):
    x = p1[0]-p2[0]
    y = p1[1]-p2[1]
    return(NP.linalg.norm(NP.array([x,y])))

def findCrossing(stk1,ind1,stk2,ind2):
    pts1 = NP.asarray(stk1.asPoints())
    pts2 = NP.asarray(stk2.asPoints())
    p11 = pts1[max(0,ind1-1),:]
    p12 = pts1[min(len(pts1)-1,ind1+1),:]
    p21 = pts2[max(0,ind2-1),:]
    p22 = pts2[min(len(pts2)-1,ind2+1),:]
    L1 = line(p11,p12)
    L2 = line(p21,p22)
    I = NP.asarray(intersection(L1,L2))
    if(I.all()==False):
        return False
    if(((dist(p11,I)+dist(I,p12)-dist(p11,p12))<10**-5) &\
        ((dist(p21,I)+dist(I,p22)-dist(p21,p22))<10**-5)):
            return True
    return False
    
def getCrossStroke(expr):
    l = len(expr.strokes)
    i = 0
    crossStrokes = []
    index = NP.array([])
    while(i<l):
        if(sum(index==i)>0):
            i = i+1
            continue
        stks = NP.array([i])
        t = min(i+5,l)
        for j in range(i+1,t):
            minDist,ind1,ind2 = getMinDist(expr.strokes[i],expr.strokes[j])
            if(findCrossing(expr.strokes[i],ind1,expr.strokes[j],ind2)):
                stks = NP.append(stks,j)
        index = NP.append(index,stks)
        crossStrokes.append(list(map(lambda i: expr.strokes[i],stks)))
        i = i+1
    return(crossStrokes)

#exprs , classes= SymbolData.unpickleSymbols("train.dat")
#symbols = SymbolData.allSymbols(exprs)
#scale = 29
#symbols = SymbolData.normalize(symbols,scale)

#exprs = SymbolData.readInkmlDirectory('inkml','lg')
#expr = exprs[10]
#plt.figure()
#expr.plot()
#crossStrokes = getCrossStroke(expr)

# Based on crossing
#l = len(expr.strokes)
#i = 0
#crossStrokes = []
#index = NP.array([])
#while(i<l):
#    if(sum(index==i)>0):
#        i = i+1
#        continue
#    stks = NP.array([i])
#    t = min(i+5,l)
#    for j in range(i+1,t):
#        minDist,ind1,ind2 = getMinDist(expr.strokes[i],expr.strokes[j])
#        if(findCrossing(expr.strokes[i],ind1,expr.strokes[j],ind2)):
#            stks = NP.append(stks,j)
#    index = NP.append(index,stks)
#    crossStrokes.append(list(map(lambda i: expr.strokes[i],stks)))
#    i = i+1

# Based on bounding box
#l = len(expr.strokes)
#i = 0
#crossStrokes = []
#index = NP.array([])
#while(i<l):
#    if(sum(index==i)>0):
#        i = i+1
#        continue
#    stks = NP.array([i])
#    bbox1 = SymbolData.getBBoxStorke(expr.strokes[i])
#    t = min(i+5,l)
#    for j in range(i+1,t):
#        bbox2 = SymbolData.getBBoxStorke(expr.strokes[j])
#        if(doBboxOverlap(bbox1,bbox2) & doBboxContain(bbox1,bbox2)):
#            stks = NP.append(stks,j)
#    index = NP.append(index,stks)
#    crossStrokes.append(list(map(lambda i: expr.strokes[i],stks)))
#    i = i+1

# Based on minimum distance, mean distance and bounding box
#l = len(expr.strokes)
#i = 0
#crossStrokes = []
#index = NP.array([])
#while(i<l):
#    if(sum(index==i)>0):
#        i = i+1
#        continue
#    stks = NP.array([i])
#    bbox1 = SymbolData.getBBoxStorke(expr.strokes[i])
#    t = min(i+5,l)
#    for j in range(i+1,t):
#        bbox2 = SymbolData.getBBoxStorke(expr.strokes[j])
#        if(doBboxOverlap(bbox1,bbox2) & doBboxContain(bbox1,bbox2)):
#            minDist = getMinDist(expr.strokes[i],expr.strokes[j])
#            meanDist = getMeanDist(expr.strokes[i],expr.strokes[j])
#            sumPeri = getPerimeter(expr.strokes[i]) + getPerimeter(expr.strokes[j])
#            bBoxMeanDist = getBboxMeanDist(bbox1,bbox2)
#            if((minDist<0.1*meanDist) | (meanDist<0.1*sumPeri) | (bBoxMeanDist<0.1*sumPeri)):
#                stks = NP.append(stks,j)
#    index = NP.append(index,stks)
#    crossStrokes.append(list(map(lambda i: expr.strokes[i],stks)))
#    i = i+1

#i=0
#for symbol in symbols:
#    print(i)
#    I = Features.symbolFeatures(symbol)
#    i+=1

train.main(['-rf', 'RF20_MaxDepth.mdl', 'train', 'trainLg'])
