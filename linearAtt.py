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

SboxNum = 1
TRACES_NUMBER = 100
TRACE_LENGTH = 700 ## number of samples
TRACE_STARTING_SAMPLE = 0
offset = 0
KNOW_KEY = b'\x2b\x7e\x15\x16\x28\xae\xd2\xa6\xab\xf7\x15\x88\x09\xcf\x4f\x3c'
ALL_POSSIBLE_KEY = int("0xff", 16)

traceRange = range(0, TRACES_NUMBER)
sampleRange = (TRACE_STARTING_SAMPLE, TRACE_STARTING_SAMPLE + TRACE_LENGTH)
# load traces, data
npzfile = np.load('traces/swaes_atmega_power.trs.npz')
traces = npzfile['traces'][offset:offset + TRACES_NUMBER,sampleRange[0]:sampleRange[1]]

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

def attakSbox(SboxNum, attackModel):

    # infoNpzFile(npzfile)
    data = npzfile['data'][:, SboxNum]
    (numTraces, traceLength) = traces.shape
    CondAver = ConditionalAveragerAesSbox(256, traceLength)

    for i in traceRange: #200 trace attack
        CondAver.addTrace(data[i], traces[i])


    (avdata, avtraces) = CondAver.getSnapshot()

# rez = lraAES(avdata, avtraces, SboxNum, basisModelSingleBits)

    rez = attackMoreMoreFaster(avdata, avtraces, SboxNum, attackModel)

    # file = open('fastfast', 'w')
    # file.write(str(rez))
    # file.write('\n')
    # file.close()
    analyzeTool_top5(rez)

    sys.exit()
    int_LraWinCandidate = int(LraWinningCandidate, 16)
    plt.plot(rez.T, color='gray')
    print("key correct: ", KNOW_KEY[SboxNum])
    print("key candidate: ", int_LraWinCandidate)

    if int_LraWinCandidate != KNOW_KEY[SboxNum]:
        plt.plot(rez[int_LraWinCandidate, :], 'blue')
        pass

# plt.xlabel("Years of experience") # adding the name of x-axis
# plt.ylabel("Salaries") # adding the name of y-axis
    plt.plot(rez[KNOW_KEY[SboxNum], :], 'r')
    plt.show()


    return  (LraWinningCandidate, LraWinningCandidatePeak)

# startTime = datetime.now()
# keyfinal = []
# for attNum in range(0,1):
#     keyfinal.append(attakSbox(attNum))
# print(keyfinal)
# print("TIME: ", datetime.now()- startTime)

attackModel = basisModelHW
attakSbox(2, attackModel)




