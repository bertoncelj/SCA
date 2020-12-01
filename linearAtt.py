import sys
import numpy as np
import matplotlib.pyplot as plt
from  numpy import linalg as LA
import statsmodels.api as sm

from lraAttack import *
from condaveraes import * # incremental conditional averaging
import functools
import operator
from datetime import datetime

# print whole np array
np.set_printoptions(threshold=sys.maxsize)
# Questions :
    # Why does data[:, SboxNum] takes all N of traces but traces dont?
    # Why does offset in traces fuck up everythink?

####################### CONFIG ############################ 
SboxNum = 1
TRACES_NUMBER = 100
TRACE_LENGTH = 400 ## number of samples
TRACE_STARTING_SAMPLE = 1000
offset = 0
traceRoundNumber=30

KNOW_KEY = b'\x2b\x7e\x15\x16\x28\xae\xd2\xa6\xab\xf7\x15\x88\x09\xcf\x4f\x3c'
ALL_POSSIBLE_KEY = int("0xff", 16)
####################### CONFIG ############################ 

# Load files -> pick out traces and data
traceRange = range(0, TRACES_NUMBER)
sampleRange = (TRACE_STARTING_SAMPLE, TRACE_STARTING_SAMPLE + TRACE_LENGTH)

npzfile = np.load('traces/swaes_atmega_power.trs.npz')
traces = npzfile['traces'][offset:offset + TRACES_NUMBER,sampleRange[0]:sampleRange[1]]
data = npzfile['data'][:, SboxNum]

def saveResultIntoFile(fileName, dataToSave):
    file = open('fastfast', 'w')
    file.write(str(rez))
    file.write('\n')
    file.close()
    return True

def infoNpzFile(npzfile):
    allNpzTypes = ('data', 'traces')
    for typeNpz in allNpzTypes:
        # construct matrix
        colums = len(npzfile[typeNpz][:,0])
        rows = len(npzfile[typeNpz][0,:])
        print("Npzfile: ", typeNpz)
        print("Matrix size: rows:", rows, " colums: ", colums)

def printTrace(npzTrace, numTrace=0):
    xLength = len(npzTrace[numTrace, :])
    print("len x: ", xLength)
    xSamples = range(0, xLength)
    print("x: ", xSamples)
    ySamples = npzTrace[numTrace, :]
    plt.plot(xSamples, ySamples, color='red')

    plt.xlabel("Time") # adding the name of x-axis
    plt.ylabel("Y samples") # adding the name of y-axis
    plt.show() # specifies end of graph

def attackSbox(avgData, avgTraces, SboxNum, attackModel):

    rez = attackMoreMoreFaster(avgData, avgTraces, SboxNum, attackModel)
    printWinningKey(rez,KNOW_KEY[SboxNum])
    # displayR2WinningKeys(rez, KNOW_KEY, SboxNum)

    # analyzeTool_top5(rez)
    corrPoint = getKeyLocationOnTrace(rez, avgTraces)
    print("Point: ", corrPoint)
    # displayCorrectKeyOnTrace(avtraces, corrPoint, TRACE_STARTING_SAMPLE)

    # return  (LraWinningCandidate, LraWinningCandidatePeak)
    return True

def attackAESinRounds(data, traces, traceRoundNum, traceNum, SboxNum):
    # calculate next trece to take 
    print("input traceRoundNum: ", traceRoundNum)
    print("input traceNum: ", traceNum)
    getAttRounds = round(traceNum/traceRoundNum)
    print("getAttR: ", getAttRounds)

    (numTraces, traceLength) = traces.shape

    Avg = ConditionalAveragerAesSbox(256, traceLength)

    trArr = [x * traceRoundNum for x in range(0,getAttRounds+1)]

    tracePairs = list(zip(trArr, trArr[1:]))

    for index, span in enumerate(tracePairs):
        for i in range(span[0], span[1]):
            Avg.addTrace(data[i], traces[i])

        (avgData, avgTraces) = Avg.getSnapshot()
        # print("avgData: ", avgData)
        # print("avgTraces: ", avgTraces)
        print("")
        print("#"*20)
        print("(" , index+1,"/",len(trArr),") Attack by blocks")
        print("Trace attack to: N = ", span[1])

        attackModel = basisModelSingleBits
        attackSbox(avgData, avgTraces, SboxNum, attackModel)


########## START ##########
if __name__== "__main__":

    attackAESinRounds(data, traces, traceRoundNumber , TRACES_NUMBER, SboxNum)
    sys.exit()

