# -*- coding: utf-8 -*-
"""
Created on Wed May 20 19:28:00 2015

@author: sxd7257
"""

import sys
import pickle
import SymbolData
#import Classification
import Features
#from sklearn.metrics import accuracy_score

usage = "Usage: $ python trainExprs.py outFilename(modelFilename) inkmldir lgdir"

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:] #dirty trick to make this convenient in the interpreter.
    if (len (argv) < 3 or len (argv) > 4): 
        print(("bad number of args:" , len(argv)))
        print (usage)
    else:
        exprs = SymbolData.readInkmlDirectory(argv[2], argv[3],True,True)
        scale = 299
        exprs = SymbolData.normalizeExprs(exprs,scale)
        
        file = open(argv[1],'wb')
        store = []
        for expr in exprs:
            if(len(expr.symbols)==19):
                I = Features.getImgExpr(expr)
                store.append([expr, I, len(expr.symbols)])
        pickle.dump(store,file,pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    sys.exit(main())