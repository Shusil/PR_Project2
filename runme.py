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


#exprs , classes= SymbolData.unpickleSymbols("train.dat")
#symbols = SymbolData.allSymbols(exprs)
#scale = 29
#symbols = SymbolData.normalize(symbols,scale)


exprs = SymbolData.readInkmlDirectory('inkmlTest','lgTest')
for expr in exprs:
#    plt.figure()
    expr.plot()
    wait = input("press enter")
#crossStrokes = getCrossStroke(expr)

#expr = SymbolData.readInkml('inkml_test/65_Frank.inkml','lg_test')
#plt.figure()
#expr.plot()

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

#train.main(['-rf', 'RF20_MaxDepth.mdl', 'train', 'trainLg'])
