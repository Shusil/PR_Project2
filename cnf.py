# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:09:59 2015

@author: bencoh
"""
import pprint
from tabulate import tabulate
pp = pprint.PrettyPrinter(indent=4)
symbols = open("symbols.txt")
symbList = set()
for line in symbols:
    symbList.add(line.strip())
print(symbList)

grammar = open('grammar.txt')
productions = {}
start = True
for line in grammar:
    if start:
        line = line.strip()

        line = line.split('->')
        left = line[0]
#        print("ADDING RULE FOR", left)
        right = line[1].split(' ')[1:]
        left = left[:-1]
        if left not in productions:
            productions[left] = []
        productions[left].append(right)
        start = False
    elif line == '\n':
        start = True
    elif '->' not in line:
        line = line.strip()
        line = line.split('| ')
#        print(line[1:][0].split(' '))
        productions[left].append(line[1:][0].split(' '))

#pp.pprint(productions)

newProd = {}
for left in productions:
    newProd[left] = []
    count = 0
    for production in productions[left]:
        if len(production) <= 2:
            newProd[left].append(production)
        else:
            while(len(production) > 2):
                newProd[left].append([production[0], left+'_'+str(count)])
                count += 1
                production = production[1:]
            
pp.pprint(newProd)   


#sent = '(2+2)=A'
sent='2+2'
def  getProdRules(symbol, grammar):
    if type(symbol) != type([]):
        return [x for x in grammar if [symbol] in grammar[x]]
    else:
        return [x for x in grammar if symbol in grammar[x]]
    
    
tokens = list(sent)
#print(tokens)    
n = len(tokens) 
table = []
for x in range(n):
    tmp = []
    for y in range(n):
        tmp.append([])
    table.append(tmp)
print(tabulate(table))

for i in range(n):
    for var in tokens:
        if var == tokens[i]:
            prodRule = getProdRules(var, newProd)
#            table[i][i] = [prodRule[0]]
            table[i][i] = [prodRule[0]]
#            
#for d in range(2,n):
#    print("D: ", d)
#    for i in range(n-d):
#        print("I: ", i)
#        for k in range(i+1,i+d):
#            print("LOOKING AT", i,' ',k)
#            for A in tokens:
#                print(table[i][k])
#                for B in table[i][k]:
##                    print(k, d)
#                    for C in table[k][k+d]:
#                        print("CONSIDERING", b, c)
#                        if A in getProdRules([B,C]):
#                            table[i][k+d] = A
            
#for x in range()
print(tabulate(table))

#print(getProdRules(["SUP", 'TERM_2'], newProd))
#gramString = "S -> FORMULA\n"
#for left in newProd:
#    gramString += left + ' -> '
#
#    for production in newProd[left]:
#        
##        production = ["'" + x + "'" if (x in symbList or (not x.isalpha() and '_' not in x)) else x for x in production ]
#        production = ["'" + x + "'" if ((not x.isalpha() and '_' not in x)) else x for x in production ]
#        
#        gramString += ' '.join(production) + " | "
#    gramString = gramString[:-2]
#    gramString += '\n'
#
#print(gramString)
#from nltk import CFG
#grammar = CFG.fromstring(gramString)
#print(grammar)
##print(len(list(generate(grammar, depth=6))))
#import nltk
#parser = nltk.ChartParser(grammar)
#
#for tree in parser.parse("24"):
#    print("TREE")
#    print(tree)
#    tree.draw()
#print(grammar.start())
#from nltk.parse.generate import generate
#for g in generate(grammar, n=5, depth=15):
#    print(g)
#print(productions['INTEGRATION '])