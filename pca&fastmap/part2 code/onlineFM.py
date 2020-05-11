#
# Group member:
# Ziqiao Gao 2157371827
# He Chang 5670527576
# Fanlin Qin 5317973858
# citation: https://github.com/anupamish/FastMap/blob/master/FastMap.py
#


import numpy as np
import math
from sys import maxsize
from random import *
import matplotlib.pyplot as plt

file = open("fastmap-data.txt",'r')
inputs = file.read()
data=  [(line.split()) for line in inputs.split("\n")]

data = data[:-1]

mat = np.zeros((11,11))
col=0

X = np.zeros((11,3)) #ndarray for storing the output - N x k

for i in range(0,len(data)):
    curLine = data[i]
    mat[int(curLine[0])][int(curLine[1])]=int(curLine[2])
    mat[int(curLine[1])][int(curLine[0])]=int(curLine[2])

def chooseDistantObjects(recurVal):
    iterationCount = 10
    globalA = 0
    globalB = 0
    globalMax = 0
    b = randint(1,10) #this returns a random integer in the range of inclusive (1-10)
    a=-1
    for i in range(0,iterationCount):
        localMax = -1000;
        for j in range(1, 11):
            # localMax = -sys.maxsize-1;
            dist = distance(b, j, recurVal)
            if dist > localMax:
                localMax = dist
                a = j

# If the furthest pair of objects is not unique, please use the one that includes the
# smallest object ID.
        if localMax>globalMax:
            globalMax=localMax
            globalA = a
            globalB = b
        elif localMax==globalMax:
            cur_min = min(a,b)
            g_min = min(globalA,globalB)
            if cur_min<g_min:
                globalA = a
                globalB = b
        b=a
    # Return the point with lower id value as origin.
    return (globalA,globalB) if globalA<globalB else (globalB,globalA)

def distance(a,b,col):
    if col==1:
        return mat[a][b]
    else:
        return math.pow((math.pow(distance(a,b,col-1),2) - math.pow((X[a][col-1]- X[b][col-1]),2)),0.5)

def FastMap(k):
    global col
    if k<1:
        return
    else:
        col+=1
    #col will be used in two places
    #1. at anytime determine the recursion stack level count
    #2. At anytime determine which X column the attribute values will be assigned to
    #get index values of the distant objects
    a,b = chooseDistantObjects(col)
    print("Iteration {}".format(col))
    print ("A val:",a)
    print ("B val:",b)
    #If distance between a and b is zero then the whole vector of that dimension(recursion) goes to 0.
    if distance(a,b,col)==0:
        X[:,col]=0 #This sets whole vector of that dimension equal to zero
    for i in range(1,11):
        if i==a:
            X[i,col]=0
        elif i==b:
            X[i,col]=distance(a,b,col)
        else:
            X[i,col] = (float)((math.pow(distance(a,i,col),2) + math.pow(distance(a,b,col),2) - math.pow(distance(b,i,col),2)))/(2*distance(a,b,col))

    FastMap(k-1)

FastMap(2)

print ("-----------------------FINAL X ndarray -  N x K arary-----------------------------")
print(X[1:,1:])
wordsToMap = []
f = open('fastmap-wordlist.txt')
line = f.readline()
##initialize 2-D arrays
while line:
    wordsToMap.append(line.splitlines())
    line = f.readline()
f.close()


X = [X[i][1:] for i in range(1, 11)]  ##Stripping the first column and the first row


# Plot the points
fig, ax = plt.subplots()
for i in range(10):
    ax.scatter(X[i][0], X[i][1])
    ax.annotate(wordsToMap[i], (X[i][0], X[i][1]))
plt.show()