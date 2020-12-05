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
SboxNum = 0
TRACES_NUMBER = 50
TRACE_LENGTH = 400 ## number of samples
TRACE_STARTING_SAMPLE = 0
offset = 0
traceRoundNumber=50
# PATH_TRACE = "aestraces/aes128_sb_ciph_0fec9ca47fb2f2fd4df14dcb93aa4967.trs.npz"
# PATH_TRACE = "aestraces/aes128_sb_ciph_0fec9ca47fb2f2fd4df14dcb93aa4967.trs.npz"

KNOW_KEY = b''
ALL_POSSIBLE_KEY = int("0xff", 16)

# PATH_TRACE="aes256_sb_ciph_be947018518aadeccacd0a94a3057a90c29eae7296a5ee0850e9de3db91e7d83.trs.npz"
# PATH_TRACE="traces/aes192_sb_ciph_ec40554bf67c9655d85cfd69ac04012f7ee1340ccf7b24fd.trs.npz"
PATH_TRACE="aestraces/aes128_sb_eqinvciph_f50f5782bac97baabdbe69c6cf94d2e2.trs.npz"


# KNOW_KEY = b'\x2b\x7e\x15\x16\x28\xae\xd2\xa6\xab\xf7\x15\x88\x09\xcf\x4f\x3c'
# PATH_TRACE="aes_atmega_power.trs.npz"

####################### CONFIG ############################ 

# Load files -> pick out traces and data
traceRange = range(0, TRACES_NUMBER)
sampleRange = (TRACE_STARTING_SAMPLE, TRACE_STARTING_SAMPLE + TRACE_LENGTH)

npzfile = np.load(PATH_TRACE)
traces = npzfile['traces'][offset:offset + TRACES_NUMBER,sampleRange[0]:sampleRange[1]]
data = npzfile['data'][:, SboxNum]


def findLastUnderScore(num, last):
    num = PATH_TRACE.find("_",num)
    if num == -1:
        return last
    else:
        last = num
        return findLastUnderScore(num+1, last)

def getCorrectKeyByName(name):
    cpFrom = findLastUnderScore(0,0)+1
    cpTo = PATH_TRACE.find(".trs")

    keyStr = str(PATH_TRACE[cpFrom:cpTo])
    tupKey = (list(zip(keyStr[::2], keyStr[1::2])))
    nn = [x[0]+x[1] for x in tupKey]
    print(nn)
    stToHex = lambda x: hex(int(x,16))
    stToInt = lambda x: int(x,16)
    keyHex = list(map(stToHex , nn))
    keyInt = list(map(stToInt , nn))
    KNOW_KEY = keyHex
    knownKey = np.array(keyInt, dtype="uint8")

    print("Know key: ", knownKey)
    # return keyHex
    return knownKey

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
    maxPairs = analyzeTool_top5(rez)
    dispayTop5(maxPairs)
    printWinningKey(rez,KNOW_KEY[SboxNum])
    displayR2WinningKeys(rez, KNOW_KEY[SboxNum])

    # analyzeTool_top5(rez)
    # corrPoint = getKeyLocationOnTrace(rez, avgTraces)
    # print("Point: ", corrPoint)
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
    KNOW_KEY = getCorrectKeyByName(PATH_TRACE)
    infoNpzFile(npzfile)
    attackAESinRounds(data, traces, traceRoundNumber , TRACES_NUMBER, SboxNum)
    sys.exit()

