import numpy as np
import matplotlib.pyplot as plt
from  numpy import linalg as LA

TRACE_LENGTH = 0

# For faster calculation 
byteHammingWeight = np.load('data/bytehammingweight.npy') # HW of a byte
sbox = np.load('data/aessbox.npy')

def sBoxOut(data, keyByte):
    sBoxIn = data ^ keyByte
    return sbox[sBoxIn]

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
    g.append(1)
    return g

def basisModelHW(x):
    g = []
    hw = byteHammingWeight[x]  # this is the definition: gi = HW(x)
    g.append(hw)
    g.append(1)
    return g

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

def calcRsquareFast(timeSliceTrace, appliedModelData, beta):
    E = np.dot(appliedModelData, beta)
    return  np.sum((E - timeSliceTrace) ** 2)

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


def attackMoreFaster(data, traces):
    # calculate all prediction 
    keyDataPredictions= []
    for predictionKey in range(0, ALL_POSSIBLE_KEY):
        keyByte = np.uint8(predictionKey)
        # print("key: ", keyByte)
        sBoxOut = sbox[data ^ keyByte]
        keyDataPredictions.append(sBoxOut)
    # print(keyDataPredictions)

    SStot = np.sum((traces - np.mean(traces, 0)) ** 2, 0)
    SSres= np.empty((256, TRACE_LENGTH)) # Sum of Squares due to regression
# ATTACK time slices
    # for key in range(0, ALL_POSSIBLE_KEY):
    for key in range(0, ALL_POSSIBLE_KEY):
        appliedModelData = list(map(basisModelSingleBits, keyDataPredictions[key]))
        X = np.asarray(appliedModelData)
        step1_beta = getBeta_step1(X)
        for positionTraceSample in range(0, TRACE_LENGTH):
            # print("trace sample : ", positionTraceSample)
            # print(traces[:, positionTraceSample])
            traceToAtt = traces[:, positionTraceSample]

            # print("beta in : ", keyDataPredictions[0])
            # print("beta in : ", traceToAtt)
            beta = getBeta_step2(X, traceToAtt, step1_beta)
            res = calcRsquareFast(traceToAtt, appliedModelData, beta)
            SSres[key, positionTraceSample] = res

    R2 = 1 - SSres/ SStot[None, :]

    return R2

def attackMoreMoreFaster(data, traces,SboxNum, intermediateFunction):
    # calculate all prediction 
    keyDataPredictions= []
    for key in np.arange(0, 256, dtype='uint8'):
        sBoxOut = sbox[data ^ key]
        keyDataPredictions.append(sBoxOut)

    (numTraces, traceLength) = traces.shape
    # For R2
    SStot = np.sum((traces - np.mean(traces, 0)) ** 2, 0)

    # reserce space
    SSres= np.empty((256, traceLength)) # Sum of Squares due to regression

    # ATTACK time slices
    for key in np.arange(0, 256, dtype='uint8'):

        # Data map over preciston fucntion
        X = np.array(list(map(intermediateFunction, keyDataPredictions[key])))

        #beta part calculated
        leva = np.linalg.inv(np.dot(X.T, X))

        for positionTraceSample in range(0, traceLength):

            # take same time sample over all traces
            timeSliceTrace = traces[:, positionTraceSample]

            # beta other half calculated
            desna = np.dot(X.T, timeSliceTrace)
            beta = np.dot(leva, desna)

            E = np.dot(X, beta)
            res =  np.sum((E - timeSliceTrace) ** 2)
            SSres[key, positionTraceSample] = res

    # final list R2 for every sample for all keys
    # R2: dim (256, TRACE_LENGTH)
    R2 = 1 - SSres/ SStot[None, :]

    return R2

# TODO parametrize hard-coded values such as 256, 8
def lraAES(data, traces, intermediateFunction, basisFunctionsModel):

    ### 0. some helper variables
    (numTraces, traceLength) = traces.shape

    # define a wrapper for currying (incremental parameter binding)
    def basisFunctionsModelWrapper(y):
        def basisFunctionsModelCurry(x):
            return basisFunctionsModel(x, y)
        return basisFunctionsModelCurry

    ### 1: compute SST over the traces
    SStot = np.sum((traces - np.mean(traces, 0)) ** 2, 0)

    ### 2. The main attack loop

    # preallocate arrays
    SSreg = np.empty((256, traceLength)) # Sum of Squares due to regression
    E = np.empty(numTraces)              # expected values

    allCoefs = [] # placeholder for regression coefficients

    # per-keycandidate loop
    for k in np.arange(0, 256, dtype='uint8'):
        # predict intermediate variable

        intermediateVariable = sBoxOut(data, k)

        # buld equation system
        M = np.array(list(map(basisModelSingleBits, intermediateVariable)))

        # some precomputations before the per-sample loop
        P = np.dot(np.linalg.inv(np.dot(M.T, M)), M.T)
        #Q = np.dot(M, P)

        coefs = [] # placeholder for regression coefficients

        # per-sample loop: solve the system for each time moment
        for u in range(0,traceLength):

            # if do not need coefficients beta - use precomputed value
            #np.dot(Q, traces[:,u], out=E)

            # if need the coefficients - do the multiplication using
            # two dot products and let the functuion return beta alongside R2
            beta = np.dot(P, traces[:,u])
            coefs.append(beta)
            E = np.dot(M, beta)

            SSreg[k,u] = np.sum((E - traces[:,u]) ** 2)

        allCoefs.append(coefs)
        #print 'Done with candidate', k

    ### 3. compute Rsquared
    R2 = 1 - SSreg / SStot[None, :]

    return R2

# mozn da mam napko v attackMoreMoreFast
# probi lraAES

def analyzeTool_top5(R2outputs):
    print(R2outputs)

    maxLine = np.amax(R2outputs, axis=1)
    test = maxLine
    temp = np.argpartition(-test, 4)
    result_args = temp[:4]

    temp = np.partition(-test, 4)
    result = -temp[:4]

    print("maxLine: ", maxLine)
    print("result_args: ", result_args)
    print("result: ", result) 
    LraWinningCandidate = hex(np.argmax(maxLine))
    LraWinningCandidatePeak = np.max(maxLine)
    MaxVKey = np.where(maxLine == LraWinningCandidatePeak)[0]
    MaxSample = np.where(rez == LraWinningCandidatePeak)[0]
    print(MaxVKey)
    print(MaxSample)
    print("maxLine: ", maxLine)
    print("LRA: ", LraWinningCandidate)
    print("LRAPeak: ", LraWinningCandidatePeak)



