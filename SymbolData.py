import xml.etree.ElementTree as ET
import matplotlib as MP
import numpy as NP
import matplotlib.pyplot as PLT
from pylab import *
import os
import shutil
import re
import random
import numpy.random
import scipy.stats
from pandas import *
import pickle
import functools
from scipy.spatial import distance
import Classification
import itertools
from functools import reduce
import copy
import random


""" Contains representations for the relevant data,
    As well as functions for reading and processing it. """

class Stroke:
    """Represents a stroke as an n by 2 matrix, with the rows of
      the matrix equivelent to points from first to last. """
    def __init__(self, points, flip=False, ident=None, smooth=True, resample=True, npts=30):
        self.ident = ident
        self.xs = []
        self.ys = []
#        print('prenorm', points)
#        if(smooth):
#            points = DataFrame(points).drop_duplicates().values
#            points = smoothPoints(points)
#        if(resample):
#            points = resamplePoints(points,npts)
        for point in points:
            self.addPoint(point, flip)
#        print('postnorm', self.xs)

    def plot(self, show = True, clear = True):
        if clear:
           PLT.clf()
        PLT.plot(self.xs, self.ys, 'ko-' )
           
        if show:
            PLT.show()

    def addPoint(self, point, flip = False):
        self.xs.append(point[0])
        if flip:
            self.ys.append(-1 * point[1])
        else:
            self.ys.append(point[1])
#            self.ys[i] = (self.ys[i-1]+self.ys[i]+self.ys[i+1])/3
            
    def asPoints(self):
        return (list(zip(self.xs, self.ys)))

    def scale(self, xmin, xmax, ymin, ymax, xscale, yscale):
        if (xmax != xmin):
            self.xs = list(map( (lambda x: xscale * ((x - xmin) * 1.0 / (xmax - xmin))), self.xs))
        else:
            self.xs = list(map( (lambda x: 0), self.xs))
        if (ymax != ymin):
            self.ys = list(map( (lambda y: yscale * ((y - ymin) * 1.0 / (ymax - ymin))), self.ys))
        else:
            self.ys = list(map( (lambda y: 0), self.ys))
        self.xs = list(map( (lambda x: (x * 2) - xscale), self.xs))
        self.ys = list(map( (lambda y: (y * 2) - yscale), self.ys))

    def xmin(self):
        return min(self.xs)

    def xmax(self):
        return max(self.xs)

    def ymin(self):
        return min(self.ys)

    def ymax(self):
        return max(self.ys)
        
    def __str__(self):
        return 'Stroke:\n' + str(self.asPoints())

    
    
class Symbol:
    """Represents a symbol as a list of strokes. """
    def __init__(self, strokes, correctClass = None, norm = False, ident = None):
        assert strokes != None
        self.strokes = strokes
        self.ident = ident
        if norm:
            self.normalize()
        self.correctClass = correctClass

    def plot(self, show = True, clear = True):
        if clear:
            PLT.clf()
        for stroke in self.strokes:
            stroke.plot(show = False, clear = False)
        if show:
            PLT.show()

    def xmin(self):
        return min(list(map( (lambda stroke: stroke.xmin()), self.strokes)))

    def xmax(self):
        return max(list(map( (lambda stroke: stroke.xmax()), self.strokes)))

    def ymin(self):
        return min(list(map( (lambda stroke: stroke.ymin()), self.strokes)))

    def ymax(self):
        return max(list(map( (lambda stroke: stroke.ymax()), self.strokes)))

    def points(self):
        return functools.reduce( (lambda a, b : a + b), (list(map ((lambda f: f.asPoints()), self.strokes))), [])

    def xs(self):
        return functools.reduce( (lambda a, b : a + b), (list(map ((lambda f: f.xs), self.strokes))), [])

    def ys(self):
        return functools.reduce( (lambda a, b : a + b), (list(map ((lambda f: f.ys), self.strokes))), [])
    
    def center(self):
        return [(self.xmax() + self.xmin())/2, (self.ymax() + self.ymin())/2]
        
    def normalize(self):

        self.xscale = 1.0
        self.yscale = 1.0
        self.xdif = self.xmax() - self.xmin()
        self.ydif = self.ymax() - self.ymin()
        #look out for a divide by zero here.
        #Would fix it, but still not quite sure what the propper way to handel it is.
        if (self.xdif > self.ydif):
            self.yscale = (self.ydif * 1.0) / self.xdif
        elif (self.ydif > self.xdif):
            self.xscale = (self.xdif * 1.0) / self.ydif

        self.myxmin = self.xmin()
        self.myxmax = self.xmax()
        self.myymin = self.ymin()
        self.myymax = self.ymax()
        
        for stroke in self.strokes:
            stroke.scale(self.myxmin, self.myxmax, self.myymin, self.myymax, self.xscale, self.yscale)

    # Given a class, this produces lines for an lg file.
    def lgline(self, clss):
#        print("STROKES", self.strokes)
        self.line = 'O, ' + self.ident + ', ' + clss + ', 1.0, ' + (', '.join(list(map((lambda s: str(s.ident)), self.strokes)))) + '\n'
        #do we need a newline here? Return to this if so.        
        return self.line
            
    def __str__(self):
        self.strng = 'Symbol'
        if self.correctClass != '' and self.correctClass != None:
            self.strng = self.strng + ' of class ' + self.correctClass
        self.strng = self.strng + ':\n Strokes:'
        for stroke in self.strokes:
            self.strng = self.strng + '\n' + str(stroke)
        return self.strng
    


# Holds the symbols from an inkml file.
class Expression:

    def __init__(self, name, symbols, relations, norm = True):
        self.name = name
        self.symbols = symbols
#        symbols[0].plot()
#        self.strokes = functools.reduce((lambda a,b: a+b), list(map((lambda symbol: symbol.strokes), self.symbols)))
        self.strokes = []
#        print("CONSTRUCTING")
#        print(symbols)
        for s in symbols:
            self.strokes.extend(s.strokes)
#            print(s.strokes)
#        print(self.strokes)
        self.relations = relations
#        print(relations)
        if norm:
            self.normalize()
        self.classes = []

    def plot(self, show = True, clear = True):
        if clear:
            PLT.clf()
        
        for stroke in self.strokes:
            stroke.plot(show = False, clear = False)
        if show:
            PLT.show()

    def xmin(self):
#        return -1
#        print(self.strokes)
#        return 0
        tmp = list(map( (lambda stroke: stroke.xmin()), self.strokes))
#        print(tmp)
        return min(tmp)

    def xmax(self):
#        print("In XMAX")
#        return 0
#        print(self.strokes)
        return max(list(map( (lambda stroke: stroke.xmax()), self.strokes)))

    def ymin(self):
        return min(list(map( (lambda stroke: stroke.ymin()), self.strokes)))

    def ymax(self):
        return max(list(map( (lambda stroke: stroke.ymax()), self.strokes)))

    def points(self):
        return functools.reduce( (lambda a, b : a + b), (list(map ((lambda f: f.asPoints()), self.strokes))), [])

    def xs(self):
        return functools.reduce( (lambda a, b : a + b), (list(map ((lambda f: f.xs), self.strokes))), [])

    def ys(self):
        return functools.reduce( (lambda a, b : a + b), (list(map ((lambda f: f.ys), self.strokes))), [])
    
    def normalize(self):
#        return
#        for s in self.symbols:
#            print(s)
        self.xscale = 1.0
        self.yscale = 1.0
        self.xdif = self.xmax() - self.xmin()
        self.ydif = self.ymax() - self.ymin()
        #look out for a divide by zero here.
        #Would fix it, but still not quite sure what the propper way to handel it is.
        if (self.xdif > self.ydif):
            self.yscale = (self.ydif * 1.0) / self.xdif
        elif (self.ydif > self.xdif):
            self.xscale = (self.xdif * 1.0) / self.ydif

        self.myxmin = self.xmin()
        self.myxmax = self.xmax()
        self.myymin = self.ymin()
        self.myymax = self.ymax()
        
        for stroke in self.strokes:
            stroke.scale(self.myxmin, self.myxmax, self.myymin, self.myymax, self.xscale, self.yscale)

    def writeLG (self, directory, clss = None):
        self.filename = os.path.join(directory, (self.name + '.lg'))
        if (clss == None):
            print ("none clss")
            assert (len (list(self.classes)) == len (list(self.symbols)))
            self.clss = list(self.classes)
        else:
            self.clss = list(clss)
            
        self.symblines =  []
        self.i = 0

        #for c in (self.clss):
        #    print (c)
       # print (len(self.clss ), " ", len(list(self.symbols)))
        #print (self.clss)
        #for c in (self.clss):
            #print ( c)
#        print('clss', self.clss)
#        print(len(self.symbols))
        for symbol in self.symbols:
           # print (self.i)
            self.symblines.append(symbol.lgline(self.clss[self.i]))
            self.i = self.i + 1
        
        with (open (self.filename, 'w')) as f:
            
            for line in self.symblines:
                f.write(line)

            f.write('\n#Relations imported from original\n')
            
            for relation in self.relations:
                f.write(relation)

defaultClasses = ['!', '(', ')', '+', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'A', 'B', 'C', 'COMMA', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Y', '[', '\\Delta', '\\alpha', '\\beta', '\\cos', '\\div', '\\exists', '\\forall', '\\gamma', '\\geq', '\\gt', '\\in', '\\infty', '\\int', '\\lambda', '\\ldots', '\\leq', '\\lim', '\\log', '\\lt', '\\mu', '\\neq', '\\phi', '\\pi', '\\pm', '\\prime', '\\rightarrow', '\\sigma', '\\sin', '\\sqrt', '\\sum', '\\tan', '\\theta', '\\times', '\\{', '\\}', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '|']

# This stuff is used for reading strokes and symbols from files.



def readStroke(root, strokeNum):
    strokeElem = root.find("./{http://www.w3.org/2003/InkML}trace[@id='" + repr(strokeNum) + "']")
    strokeText = strokeElem.text.strip()
    pointStrings = strokeText.split(',')
    points = list(map( (lambda s: [float(n) for n in (s.strip()).split(' ')]), pointStrings))
    return Stroke(points, flip=True, ident=strokeNum, smooth=True)#, resample=True, nPts=20)


def readStrokeNew(root, strokeNum):
    # print root, str(strokeNum)
    strokeElems = root.findall("./{http://www.w3.org/2003/InkML}trace")
    for e in strokeElems:
        if e.attrib['id'] == strokeNum:
            strokeElem = e 
    # print strokeElem
    strokeText = strokeElem.text.strip()
    pointStrings = strokeText.split(',')
#    print(pointStrings[:10])
    points = list(map( (lambda s: [float(n) for n in (s.strip()).split(' ')[:2]]), pointStrings))
    
    return Stroke(points, flip=True, ident=strokeNum)


#Are there any other substitutions of this type we need to make? Come back to this.
def doTruthSubs(text):
    if text == ',':
        return 'COMMA'
    else: 
        return text


def readSymbol(root, tracegroup):
    truthAnnot = tracegroup.find(".//{http://www.w3.org/2003/InkML}annotation[@type='truth']")
    identAnnot = tracegroup.find(".//{http://www.w3.org/2003/InkML}annotationXML")    
    strokeElems = tracegroup.findall('.//{http://www.w3.org/2003/InkML}traceView')
    assert( len(strokeElems) != 0)
    strokeNums = list(map( (lambda e: int(e.attrib['traceDataRef'])), strokeElems)) #ensure that all these are really ints if we have trouble.
    strokes = list(map( (lambda n: readStroke(root, n)), strokeNums))
    if (truthAnnot == None):
        truthText = None
    else:
        truthText = doTruthSubs(truthAnnot.text)
    if identAnnot == None:
            #what do we even do with this?
            #messing with lg files depends on it.
            #for the momment, give it a bogus name and continue.
        idnt = str(strokeNums).replace(', ', '_')
    else:
        idnt = identAnnot.attrib['href'].replace(',', 'COMMA')
    return Symbol(strokes, correctClass=truthText, norm=False, ident=idnt )
    
def constructTraceGroupsFromSymbols(root, traces):
    symbols = []
    count = 0
    for trace in traces:
        sID = trace.attrib['id']
        # print 'reading'
        s = readStrokeNew(root, sID)
        # print 'read'
        # print s
        s = Symbol([s], ident='x'+str(count)+'_')
        count += 1
        symbols.append(s)

    # print 'here'
    # symbols = merge(symbols)
    # print symbols

    return symbols



def readFile(filename, warn=True, option='-t'):
    if(option=='-pcs'):
        try:
            print ("Parsing", filename)
            tree = ET.parse(filename)
    #        print("PARSED")
            root = tree.getroot()
            traces = root.findall('./{http://www.w3.org/2003/InkML}trace')
    #        print("Got traces")
            tracegroups = constructTraceGroupsFromSymbols(root, traces)
    #        print("Returning", tracegroups)
            return tracegroups
        except:
            print("Unparsable file")
            if warn:
                print("warning: unparsable file -- ",filename)
            return []
    else:
        try:
            #print (filename)
            tree = ET.parse(filename)
            root = tree.getroot()
            tracegroups = root.findall('./*/{http://www.w3.org/2003/InkML}traceGroup')
            symbols = list(map((lambda t: readSymbol(root, t)), tracegroups))
            return symbols
        except:
            if warn:
                print("warning: unparsable file -- ",filename)
            return []

def fnametolg(filename, lgdir):
    fdir, fname = os.path.split(filename)
    name, ext = os.path.splitext(fname)
    return os.path.join(lgdir, (name + ".lg"))

def mergeFromCrossings(expOrig):
    exp = copy.deepcopy(expOrig)
    exp = normalizeExprs([expOrig],99)
    exp = exp[0]    
    crossings = getCrossStroke(exp)
    allSymbols = []
    count = 0
    for s in crossings:
        sym = Symbol(s, ident='x'+str(count)+'_')
        count += 1
        allSymbols.append(sym)
    e = Expression(exp.name, allSymbols, exp.relations)
    return e
    
def mergeFromRecog(e):
    l = len(e.symbols)
    potentials, potentials3 = [], []
    for x in range(l-1):
        potMerge = [e.symbols[x], e.symbols[x+1], x]
        potentials.append(potMerge)
    for x in range(l-2):
        potMerge = [e.symbols[x], e.symbols[x+1], e.symbols[x+2], x]
        potentials3.append(potMerge)

    confs = {}
    for pair in potentials:
        # local normalized stroke paris for classification
        p1 = copy.deepcopy(pair[0].strokes)
        p2 = copy.deepcopy(pair[1].strokes)
            
        s1 = Symbol(p1,norm=True)
        s2 = Symbol(p2,norm=True)

        cl1, symbolCl1 = Classification.classifySymbol(s1)
        cl2, symbolCl2 = Classification.classifySymbol(s2)
        strokes = []
        for stroke in pair[0].strokes:
            strokes.append(stroke)
        for stroke in pair[1].strokes:
            strokes.append(stroke)
        mergedSymb = Symbol(strokes, ident='m_'+str(random.randint(1,100)))
        strokesCopy = copy.deepcopy(strokes)
        sm = Symbol(strokesCopy, ident='m_' + str(random.randint(1,100)), norm=True)   # normalized for classification
        clBoth, symbolBoth = Classification.classifySymbol(sm)
        newESymbols = []
#        confs = {}
        if max(clBoth[0]) * 2 >= max(max(cl1[0]), max(cl2[0])):
            if (symbolCl1 != '.' and symbolCl2 != '.') or symbolBoth == 'i':
#                print("\n\n\n\n\n\nDOT\n\n\n\n\n\n\n")                
                confs[max(clBoth[0])] = [pair, mergedSymb]

    for pair in potentials3:
        # local normalized stroke paris for classification
        p1 = copy.deepcopy(pair[0].strokes)
        p2 = copy.deepcopy(pair[1].strokes)
        p3 = copy.deepcopy(pair[2].strokes)

        s1= Symbol(p1,norm=True)
        s2 = Symbol(p2,norm=True)
        s3 = Symbol(p3,norm=True)

        cl1, symbolCl1 = Classification.classifySymbol(s1)
        cl2, symbolCl2 = Classification.classifySymbol(s2)
        cl3, symbolCl3 = Classification.classifySymbol(s3)

        strokes = []
        for stroke in pair[0].strokes:
            strokes.append(stroke)
        for stroke in pair[1].strokes:
            strokes.append(stroke)
        for stroke in pair[2].strokes:
            strokes.append(stroke)

        mergedSymb = Symbol(strokes, ident='m_' + str(random.randint(0,100)))
        strokesCopy = copy.deepcopy(strokes)
        sm = Symbol(strokesCopy, ident='m_' + str(random.randint(0,100)), norm=True)   # normalized for classification
        clBoth, symbolBoth = Classification.classifySymbol(sm)
        newESymbols = []
#        confs = {}
        if max(clBoth[0]) * 3 >= max(max(cl1[0]), max(cl2[0]), max(cl3[0])):
            if (symbolCl1 != '.' and symbolCl2 != '.' and symbolCl3 != '.') or symbolBoth == 'i':
            
                confs[max(clBoth[0])] = [pair, mergedSymb]
    


    if len(confs.keys()) > 0:
        maxConf = max(confs.keys())
        pairToMerge, mergedSymb = confs[maxConf]

        # print("MERGING", pairToMerge[2], pairToMerge[2] + 1,  pairToMerge[2] + 2)
        # if len(pairToMerge) == 3:
        #     oneOrTwo = 2
        # else:
        #     oneOrTwo = 1
        for symbol in range(l):
            if len(pairToMerge) == 3:
                if (symbol != pairToMerge[2]) and (symbol != pairToMerge[2] + 1):
                    print("ADDING ORIG SYMBOL", symbol)
                    newESymbols.append(e.symbols[symbol])
                elif symbol == pairToMerge[2]:
                    newESymbols.append(mergedSymb)
            elif len(pairToMerge) == 4:
                if (symbol != pairToMerge[3]) and (symbol != pairToMerge[3] + 1) and (symbol != pairToMerge[3] + 2):
                    print("ADDING ORIG SYMBOL", symbol)
                    newESymbols.append(e.symbols[symbol])
                elif symbol == pairToMerge[3]:
                    newESymbols.append(mergedSymb)  
        # if oneOrTwo == 1:              
        #     if(pairToMerge[2] != (l-2)):
        #         newESymbols.append(e.symbols[l-1])
        # else:
        #     if(pairToMerge[3] != (l-3)):
        #         newESymbols.append(e.symbols[l-2])
           
        return Expression(name=e.name, symbols=newESymbols, relations=e.relations, norm=True)
    return e

def classifyRelationship(s1, s2):
    return "Right"

def euc_dist(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1] - p2[1])**2)**.5
def parse(e):
    try:
        match =  Classification.getMatchingExpression(e)[0]
        centers = {}
        for symbol in match.symbols:
            center = symbol.center()
            centers[str(center[0]) +' ' + str(center[1])] = symbol
            
        keys = [[float(x.split(' ')[0]), float(x.split(' ')[1])] for x in centers.keys()]
        
        for symb in e.symbols:
            center = symb.center()
            distances = {}
            for key in keys:
                distances[euc_dist(center, key)] = key
            closestDist = distances[min(distances.keys())]
            closest = centers[str(closestDist[0]) +' ' + str(closestDist[1])]
            symb.ident = closest.ident
            keys.remove(closestDist)
        e.relations = match.relations
#        print(match.relations)
    #    match.plot()
    #    relationships = []
    #    for index, symbol in enumerate(e.symbols[:-1]):
    #        rel = classifyRelationship(symbol, e.symbols[index + 1])
    #        relationships.append("EO, " + symbol.ident + ", " + e.symbols[index + 1].ident + ", " + rel + ", 1.0\n")
    #    e.relations = relationships
        return e
    except:
        e.relations = []
        return e
# this returns an expression class rather than just a list of symbols.

def readInkml(filename, lgdir, warn=False, option='-t'):
    symbols = readFile(filename, warn, option) #trainis the last param
    rdir, filenm = os.path.split(filename)
    name, ext = os.path.splitext(filenm)
    lgfile = fnametolg(filename, lgdir)
    if symbols == []:
        tmp = Stroke([[0,0],[1,1]])
        symbols = [Symbol([tmp], ident='y_')]
    e = Expression(name, symbols, readLG(lgfile), norm=True)

    if(option=='-pcs'):
        print("PREMERGE SYMBOLS", len(e.symbols))
        e = mergeFromCrossings(e)
        print("AfterCross SYMBOLS", len(e.symbols))
#       eNew = mergeFromRecog(e)
#       while(len(eNew.symbols) != len(e.symbols)):
#           e = copy.deepcopy(eNew)
#           eNew = mergeFromRecog(e)
        eNew = None
        while(eNew!=e):
            eNew = mergeFromRecog(e)
            e = eNew
#        print("PREMERGE SYMBOLS", len(e.symbols))
#        e = mergeFromCrossings(e)
#        print("AfterCross SYMBOLS", len(e.symbols))
        e = parse(e)
#    eNew = mergeFromRecog(e)
#    while(len(eNew.symbols) != len(e.symbols)):
#        e = copy.deepcopy(eNew)
#        eNew = mergeFromRecog(e)
#    e = mergeFromRecog(e)
#    if 'bert' in filename:
#        e.plot()
        print("AfterRecog SYMBOLS", len(e.symbols))

    if(option=='-pc'):
        e = parse(e)

    return e
    

def readLG(filename):

    with open(filename) as f:
        lines = f.readlines()

    relations = []
    for line in lines:
        if (line[0] == 'R' or line[0:2] =='EO'):
            relations.append(line)

    return relations        


def filenames(filename):
    inkmlre = re.compile('\.inkml$')
    fnames = []
    if(os.path.isdir(filename)):
        for root, dirs, files in os.walk(filename):
            for name in files:
                if(inkmlre.search(name) != None):
                    fnames.append(os.path.join(root, name))
    elif(inkmlre.search(filename) != None):
        fnames.append(filename)
    return fnames

def filepairs(filename, lgdir):
    fnames = filenames(filename)
    return list(map ((lambda f: (f, fnametolg(f, lgdir))), fnames))

def readDirectory(filename, warn=False):
    fnames = filenames(filename)
    return reduce( (lambda a, b : a + b), (list(map ((lambda f: readFile(f, warn)), fnames))), [])

def readInkmlDirectory(filename, lgdir, warn=False, option='-t'):
    fnames = filenames(filename)
    return list(map((lambda f: readInkml(f, lgdir, warn, option)), fnames))

def allSymbols(inkmls):
    return reduce( (lambda a, b: a + b), (list(map ((lambda i: i.symbols), inkmls))))

def symbsByClass(symbols):
    classes = {}
    for key in defaultClasses:
        classes[key] = []
    for symbol in symbols:
#        print (symbol)
        key = symbol.correctClass
        if (key not in classes):
            classes[key] = []
        classes[key].append(symbol)
    return classes

def symbClasses(symbols):
    keys = list(symbsByClass(symbols).keys())
    keys.sort()
    return keys

def exprClasses(inkmls):
    return symbClasses(allSymbols(inkmls))

def classNumbers(symbols, keys=None):
    if (keys == None):
        keys = list(symbsByClass(symbols).keys())
        keys.sort()
    cns = []
    for symbol in symbols:
       ct =  symbol.correctClass
       if ct==None:
           #cns.append(None)
           return None
       else:
           cns.append(keys.index(ct))
    return cns
    #return list(map((lambda symbol: keys.index(symbol.correctClass)), symbols))

#The function this is being fed to normalizes, so it doesn't matter that
#they don't sum to one.
def symbsPDF (symbols, keys=defaultClasses):
    if len(symbols) > 0:
        if isinstance(symbols[0], Expression):
            symbs = allSymbols(symbols)
        else:
            symbs = symbols
    

        clss = symbsByClass(symbs)
        counts = NP.array([len(clss[key]) for key in keys])
        return counts
    else:
        return numpy.zeros(len(keys))

def cleverSplit(fpairs, perc = (2.0/3), maxit = 100000):
    print ("reading files.")
    symbs = symbsByFPair(fpairs)


    print("constructing initial split")
    train, test = randSplit(fpairs, perc)

    print("getting initial PDFs")
    trnsymbs = NP.concatenate([symbs[t] for t in train])
    tstsymbs = NP.concatenate([symbs[t] for t in test])
    
    trainpdf = symbsPDF(trnsymbs)
    testpdf = symbsPDF(tstsymbs)
    #return (trainpdf, testpdf)
    #assert(len(trainpdf) == len(testpdf))
    print("getting initial entropy")
    entropy = scipy.stats.entropy(trainpdf, testpdf)
    print (entropy)
    count = 0
    while (count < maxit):
        #print("looping")
        i1 = random.randint(0, len(train)-1)
        i2 = random.randint(0, len(test)-1)
        
        i1_p = symbsPDF(symbs[train[i1]])
        i2_p = symbsPDF(symbs[train[i2]])
        new_trainpdf = (trainpdf - i1_p) + i2_p
        new_testpdf = (testpdf - i2_p) + i1_p

        new_entropy = scipy.stats.entropy(new_trainpdf, new_testpdf)
        if (new_entropy < entropy):
            print(new_entropy , " < " , entropy, ": swaping")
            testtmp = test[i2]
            test[i2] = train[i1]
            train[i1] = testtmp
            entropy = new_entropy
            trainpdf = new_trainpdf
            testpdf = new_testpdf

        count = count + 1

    print("split entropy: ", entropy)
    return (train, test)
    

def randSplit(items, perc = (2.0/3)):
    my_items = list(items)
    random.shuffle(my_items)
    splitnum = int(round(len(my_items) * perc))
    return (my_items[:splitnum], my_items[splitnum:])


def splitSymbols(symbols, trainPerc):
    classes = symbsByClass(symbols)
    training = []
    testing = []
    trainTarget = int(round(len(symbols) * trainPerc))
    testTarget = len(symbols) - trainTarget
    for clss, symbs in list(classes.items()):
        #consider dealing with unclassified symbols here if it is a problem.
        nsymbs = len(symbs)
        trainNum = int(round(nsymbs * trainPerc))
        random.shuffle(symbs)
        training = training + symbs[:trainNum]
        testing = testing + symbs[trainNum:]

    # Good enough unless the prof says otherwise.
    return( (training, testing))

#splits on a per-expression basis instead.
def splitExpressions(expressions, trainPerc):
    training = []
    testing = []
    exprs = expressions
    random.shuffle(exprs)
    trainNum = int(round (len(exprs) * trainPerc))
    training = training + exprs[:trainNum]
    testing = testing + exprs[trainNum:]

    #fancy stuff to ensure a good split goes here.
    #will try and add that today.

    return( (training, testing))

def symbsByFPair(fls):
    es = {}
    for fp in fls:
        #print fp 
        es[fp] = readFile(fp[0])
    return es

def splitFiles(inkmldir, lgdir, traindir, testdir, trainlg, testlg, trainPerc = (2.0 / 3.0)):

    training, testing = cleverSplit(list(filepairs(inkmldir, lgdir)))

   # fls = list(filepairs(inkmldir, lgdir))
   # random.shuffle(fls)
   # trainNum = int(round (len(fls) * trainPerc))
   # training = training + fls[:trainNum]
   # testing = testing + fls[trainNum:]

    for fpair in training:
        shutil.copy(fpair[0], traindir)
        shutil.copy(fpair[1], trainlg)

    for fpair in testing:
        shutil.copy(fpair[0], testdir)
        shutil.copy(fpair[1], testlg)
    #fancy stuff to ensure a good split goes here.
    #will try and add that today.

    return( (training, testing))

    
def pickleSymbols(symbols, filename):
    with open(filename, 'wb') as f:
        pickle.dump(symbols, f, pickle.HIGHEST_PROTOCOL)
        #note that this may cause problems if you try to unpickle with an older version.

def unpickleSymbols(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Normalize the data such that x or y -> (0,99) and maintain the aspect ratio
#def normalize(symbols,scale):
#    k=0
#    for symbol in symbols:
#        xmin = symbol.xmin()
#        ymin = symbol.ymin()
#        for i in range(len(symbol.strokes)):
#            for j in range(len(symbol.strokes[i].xs)):
#                symbol.strokes[i].xs[j] = (symbol.strokes[i].xs[j]-xmin)*scale/2
#                symbol.strokes[i].ys[j] = (symbol.strokes[i].ys[j]-ymin)*scale/2
#        symbols[k] = symbol
#        k+=1    
#    return(symbols)


def normalize(symbolsOrig,scale):
    k=0
    symbols = copy.deepcopy(symbolsOrig)
    for symbol in symbols:
        xmin = symbol.xmin()
        ymin = symbol.ymin()
        xmax = symbol.xmax()
        ymax = symbol.ymax()
        for i in range(len(symbol.strokes)):
            for j in range(len(symbol.strokes[i].xs)):
                rangeSym = max((ymax-ymin),(xmax-xmin))
                if(rangeSym!=0):
                    symbol.strokes[i].xs[j] = (symbol.strokes[i].xs[j]-xmin)*scale/rangeSym
                    symbol.strokes[i].ys[j] = (symbol.strokes[i].ys[j]-ymin)*scale/rangeSym
        symbols[k] = symbol
        k+=1    
    return(symbols)


def normalizeExprs(exprsOrig,scale):
    k=0
    exprs = copy.deepcopy(exprsOrig)
    for expr in exprs:
        xmin = expr.xmin()
        xmax = expr.xmax()
        ymin = expr.ymin()
        ymax = expr.ymax()
        for i in range(len(expr.strokes)):
            for j in range(len(expr.strokes[i].xs)):
                rangeExp = max((ymax-ymin),(xmax-xmin))
                if(rangeExp!=0):
                    expr.strokes[i].xs[j] = (expr.strokes[i].xs[j]-xmin)*scale/rangeExp
                    expr.strokes[i].ys[j] = (expr.strokes[i].ys[j]-ymin)*scale/rangeExp
        exprs[k] = expr
        k = k+1
    return(exprs)




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

def smoothPoints(points):
    l = points.shape[0]
    for i in range(1,l-1):
        points[i] = (points[i-1]+points[i]+points[i+1])/3
    return points

def resamplePoints(points,npts):
    newPts = NP.zeros((npts,2))
    n = points.shape[0]
    if(n<2):
        return(NP.repeat(points,npts,axis=0))
    L = NP.zeros((n))
    for i in range(1,n):
        L[i] = L[i-1]+NP.linalg.norm(points[i]-points[i-1])
    dist = L[n-1]/npts
    
    newPts[0] = points[0]
    j = 0
    for p in range(1,npts-1):
        while(L[j]<p*dist):
            j=j+1
        C = (p*dist-L[j-1])/(L[j]-L[j-1])
        newPts[p,0] = points[j-1,0]+(points[j,0]-points[j-1,0])*C
        newPts[p,1] = points[j-1,1]+(points[j,1]-points[j-1,1])*C
    newPts[npts-1] = points[-1]
    return(newPts)