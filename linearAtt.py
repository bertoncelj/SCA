import numpy as np
import matplotlib.pyplot as plt
from  numpy import linalg as LA
import statsmodels.api as sm

from condaveraes import * # incremental conditional averaging
import functools
import operator

SboxNum = 0
TRACES_NUMBER = 50
TRACE_LENGTH = 1000 ## number of samples
TRACE_STARTING_SAMPLE = 800
offset = 0
KNOW_KEY = b'\x2b\x7e\x15\x16\x28\xae\xd2\xa6\xab\xf7\x15\x88\x09\xcf\x4f\x3c'
ALL_POSSIBLE_KEY = int("0xff", 16)

# For faster calculation 
sbox = np.load('data/aessbox.npy')

traceRange = range(0, TRACES_NUMBER)
sampleRange = (TRACE_STARTING_SAMPLE, TRACE_STARTING_SAMPLE + TRACE_LENGTH)
# load traces, data
npzfile = np.load('traces/swaes_atmega_power.trs.npz')
data = npzfile['data'][:, SboxNum]
traces = npzfile['traces'][offset:offset + TRACES_NUMBER,sampleRange[0]:sampleRange[1]]

foldl = lambda func, acc, xs: functools.reduce(func, xs, acc)

def mean(vecArr):
    sumArr = foldl(operator.sub, 0, vecArr)
    return sumArr / vecArr.size

def tr(A):
    return [[row[i] for row in A] for i in range(len(A[0]))]

def lraAES(data, traces, sBox):
    finArr = []
    for i in range(0, TRACE_LENGTH):
        oneTrs = traces.T[i]
        yi_mean = mean(oneTrs)
        sum_samples = 0
        for sample in oneTrs:
            sum_samples = sum_samples + ((abs(sample) - yi_mean) ** 2)
        finArr.append(sum_samples)
    print(finArr)

    SStot = np.sum((traces - np.mean(traces, 0)) ** 2, 0)
    # print(traces)
    # print(np.mean(traces, 0))
    # print(SStot)
    # print(len(SStot))
    ### 2. The main attack loop

    # preallocate arrays
    SSreg = np.empty((64, traceLength)) # Sum of Squares due to regression
    E = np.empty(numTraces)              # expected values

    allCoefs = [] # placeholder for regression coefficient

    # per-keycandidate loop
    for k in np.arange(0, 64, dtype='uint8'):
        print(k)
        keyByte = np.uint8(k)
        sBoxOut = sbox[data ^ keyByte]
        X = list(map(leakageModel2, sBoxOut))

        # predict intermediate variable
        M = X

        # print(M)
        # some precomputations before the per-sample loop
        P = np.dot(np.linalg.inv(np.dot(tr(M), M)), tr(M))
        #Q = np.dot(M, P)

        coefs = [] # placeholder for regression coefficients

        # per-sample loop: solve the system for each time moment
        for u in range(0,TRACE_LENGTH):

            # if do not need coefficients beta - use precomputed value
            #np.dot(Q, traces[:,u], out=E)

            # if need the coefficients - do the multiplication using
            # two dot products and let the functuion return beta alongside R2
            beta = np.dot(P, traces[:,u])
            # print("beta: ", beta)
            # print("P: ", P)
            # print("traces[:,u] ", traces[:,u])
            coefs.append(beta)
            E = np.dot(M, beta)

            SSreg[k,u] = np.sum((E - traces[:,u]) ** 2)

        allCoefs.append(coefs)
        #print 'Done with candidate', k

    ### 3. compute Rsquared
    R2 = 1 - SSreg / SStot[None, :]
    return R2


def infoMatrixDemensions(A):
    print("Colums:", len(traces))
    print("Rows:", len(traces[0]))


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

# # leakage models:
# def leakageModel2():
#     a=np.array([6,1,5,0,2])
#     b=np.array(np.zeros((5)))
#     for i in range(5):
#         b[i] = '{:08b}'.format(a[i])

def basisModelSingleBits(x):
    g = []
    for i in range(0, 8):
        bit = (x >> i) & 1  # this is the definition: gi = [bit i of x]
        g.append(bit)
    g.append(1)
    return g

def leakageModel2(x):
    g = []
    for i in range(0, 2):
        bit = (x >> i) & 1
        g.append(bit)
    return g

def Rsquare(y, x):
    y_mean = np.mean(y)
    SStot = np.array([],dtype='float')

    SStot = np.sum((y - y_mean ) ** 2, 0)

    print(SStot)
    print(len(SStot))

    M = X
    P = np.dot(np.linalg.inv(np.dot(M.T, M)), M.T)

    beta = np.dot(P, Y)
    print("FINAL BETA: ", beta)
    e = Y - beta*X
    print("e:", e)
    SSres = LA.norm(e)
    print("SSres: ", SSres)

    Rsq = 1 - (SSres/SStot)
    print(Rsq)

def BetaCalc( data, traces):
    sample  = 225
    print("data", data)
    print("traces: ", traces)
    print("t:", traces[:,sample])
    print("t:", len(traces[:,sample]))
    print(np.mean(traces[:,0]))
    SStot = np.sum((traces[:,sample] - np.mean(traces[:,sample])) ** 2)
    print("Stot:", SStot)
    keyByte = np.uint8(KNOW_KEY[SboxNum])
    sBoxOut = sbox[data ^ keyByte]

    X = list(map(basisModelSingleBits, sBoxOut))

    Xnp = np.empty((TRACES_NUMBER,2)) # Sum of Squares due to regression
    print("X", X)
    X = np.asarray(X)
    print("Xnp", X)
    print("Xnp", X.T)
    leva = np.dot(X.T, X)
    leva = np.linalg.inv(leva)
    desna = np.dot(X.T, traces[:, sample])
    beta = np.dot(leva, desna)
    print("beta: ", beta)
    print(traces)
    print(desna)
    print(leva)
    E = np.dot(X, beta)
    SSreg = np.sum((E - traces[:,sample]) ** 2)

    print("SSreg: ", SSreg)
    R2 = 1 - (SSreg/SStot)
    print("R2: ", R2)

def getBeta_step1(X):
    # Equation:
    # Beta = (X.T * X)^(-1) * (X.T * Y)
    leva = np.linalg.inv(np.dot(X.T, X))
    return leva 

def getBeta_step2(X, Y, leva):
    # Equation:
    # Beta = (X.T * X)^(-1) * (X.T * Y)
    desna = np.dot(X.T, Y)
    beta = np.dot(leva, desna)

    return beta

# getBeta appliedModelData, trace
def getBeta(X, Y):
    # Equation:
    # Beta = (X.T * X)^(-1) * (X.T * Y)
    X = np.asarray(X)

    leva = np.linalg.inv(np.dot(X.T, X))
    desna = np.dot(X.T, Y)
    beta = np.dot(leva, desna)

    return beta

def calcRsquare(timeSliceTrace, appliedModelData, beta):
    SStot = np.sum((timeSliceTrace - np.mean(timeSliceTrace)) ** 2)
    E = np.dot(appliedModelData, beta)
    SSreg = np.sum((E - timeSliceTrace) ** 2)
    R2 = 1 - (SSreg/SStot)
    return R2

def attack(data, traces):

    # calculate all prediction 
    keyDataPredictions= []
    for predictionKey in range(0, ALL_POSSIBLE_KEY):
        keyByte = np.uint8(predictionKey)
        # print("key: ", keyByte)
        sBoxOut = sbox[data ^ keyByte]
        keyDataPredictions.append(sBoxOut)
    # print(keyDataPredictions)

    R2res= np.empty((256, TRACE_LENGTH)) # Sum of Squares due to regression
# ATTACK time slices
    # for key in range(0, ALL_POSSIBLE_KEY):
    for key in range(0, ALL_POSSIBLE_KEY):
        appliedModelData = list(map(basisModelSingleBits, keyDataPredictions[key]))
        for positionTraceSample in range(0, TRACE_LENGTH):
            # print("trace sample : ", positionTraceSample)
            # print(traces[:, positionTraceSample])
            traceToAtt = traces[:, positionTraceSample]

            # print("beta in : ", keyDataPredictions[0])
            # print("beta in : ", traceToAtt)
            beta = getBeta(appliedModelData, traceToAtt)
            R2 = calcRsquare(traceToAtt, appliedModelData, beta)
            R2res[key, positionTraceSample] = R2

    return R2res

def attackFaster(data, traces):
    # calculate all prediction 
    keyDataPredictions= []
    for predictionKey in range(0, ALL_POSSIBLE_KEY):
        keyByte = np.uint8(predictionKey)
        # print("key: ", keyByte)
        sBoxOut = sbox[data ^ keyByte]
        keyDataPredictions.append(sBoxOut)
    # print(keyDataPredictions)

    R2res= np.empty((256, TRACE_LENGTH)) # Sum of Squares due to regression
# ATTACK time slices
    # for key in range(0, ALL_POSSIBLE_KEY):
    for key in range(0, ALL_POSSIBLE_KEY):
        appliedModelData = list(map(basisModelSingleBits, keyDataPredictions[key]))
        X = np.asarray(appliedModelData)
        step1_beta = getBeta_step1(X)
        print(key)
        for positionTraceSample in range(0, TRACE_LENGTH):
            # print("trace sample : ", positionTraceSample)
            # print(traces[:, positionTraceSample])
            traceToAtt = traces[:, positionTraceSample]

            # print("beta in : ", keyDataPredictions[0])
            # print("beta in : ", traceToAtt)
            beta = getBeta_step2(X, traceToAtt, step1_beta)
            R2 = calcRsquare(traceToAtt, appliedModelData, beta)
            R2res[key, positionTraceSample] = R2

    return R2res



infoNpzFile(npzfile)
# printTrace(npzfile['traces'])

#build prediction
# keyByte = np.uint8(KNOW_KEY[0])
# sBoxOut = sbox[loadData ^ keyByte]
# print("loadData: ", loadData)
# print("sBoxOut: ", sBoxOut)

(numTraces, traceLength) = traces.shape
CondAver = ConditionalAveragerAesSbox(256, traceLength)

for i in traceRange: #200 trace attack
    CondAver.addTrace(data[i], traces[i])


(avdata, avtraces) = CondAver.getSnapshot()
print("avr_trace: ", avtraces)
print("avr_trace: ", len(avtraces[:,0]))
# BetaCalc(avdata, avtraces)
rez = attackFaster(avdata, avtraces)
maxLine = np.amax(rez, axis=1)
LraWinningCandidate = np.argmax(maxLine)
LraWinningCandidatePeak = np.max(maxLine)
MaxVKey = np.where(maxLine == LraWinningCandidatePeak)[0]
MaxSample = np.where(rez == LraWinningCandidatePeak)[0]
print(MaxVKey)
print(MaxSample)
print(rez)
print("maxLine: ", maxLine)
print("LRA: ", LraWinningCandidate)
print("LRAPeak: ", LraWinningCandidatePeak)
plt.scatter(list(range(0,TRACE_LENGTH)), rez[2,:], color='red') # plotting the observation line

plt.xlabel("Years of experience") # adding the name of x-axis
plt.ylabel("Salaries") # adding the name of y-axis
plt.show() # specifies end of graph


