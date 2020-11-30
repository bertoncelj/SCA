import numpy as np
import matplotlib.pyplot as plt
import time

from condaveraes import * # incremental conditional averaging

# AES S-box
sbox = np.load('data/aessbox.npy') 
# leakage model
def sBoxOut(data, keyByte):
    sBoxIn = data ^ keyByte
    return sbox[sBoxIn]

def byte_to_int(byteArr):
    arr = np.array([], dtype='uint8')
    for a in byteArr:
        arr = np.append(arr, a)
    return  arr.astype('uint8')


def basisModelSingleBits(x, bitWidth):
    g = []
    for i in range(0, bitWidth):
        bit = (x >> i) & 1  # this is the definition: gi = [bit i of x]
        g.append(bit)
    g.append(1)
    return g

# LRA attack on AES
# data                 - 1-D array of input bytes
# traces               - 2-D array of traces
# intermediateFunction - one of the functions like sBoxOut above in the common section 
# basisFunctionsModel  - one of the functions like basisModelSingleBits above
#                        in this section
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
        intermediateVariable = intermediateFunction(data, k)

        # buld equation system
        M = np.array(list(map(basisFunctionsModelWrapper(8), intermediateVariable)))

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

    return R2, allCoefs
# configurable parameters
tracesetFilename = "traces/swaes_atmega_power.trs.npz"
sampleRange = (1000, 1400) # range sample to attack 
N = 100 # number of traces to attack
offset = 0 # traces number to start from
evolutionStep = 10 # step of intermediate reports
SboxNum = 9 # S-box to attack, counting from 0

inputRange = range(0, 1000) # range for traces (not samples!)
SampleNum = 1025

npzfile = np.load('traces/swaes_atmega_power.trs.npz')
data = npzfile['data'][inputRange,SboxNum]
traces = npzfile['traces'][inputRange,SampleNum]

knowKey = b'\x2b\x7e\x15\x16\x28\xae\xd2\xa6\xab\xf7\x15\x88\x09\xcf\x4f\x3c'
# keyByte = np.uint8(key[SboxNum])
# sBoxOut = sbox[data ^ keyByte]


# encrypt in uint8
knownKey1 = byte_to_int(knowKey)

npzfile = np.load(tracesetFilename)
data = npzfile['data'][offset:offset + N,SboxNum] # selecting only the required byte
traces = npzfile['traces'][offset:offset + N,sampleRange[0]:sampleRange[1]]

# Log traceset parameters
(numTraces, traceLength) = traces.shape
print("Number of traces loaded :", numTraces)
print("Trace length            :", traceLength)


#######################################
### Attack LRA

# initialize the incremental averager
CondAver = ConditionalAveragerAesSbox(256, traceLength)


numSteps = int(np.ceil(N / np.double(evolutionStep)))
keyRankEvolutionLRA = np.zeros(numSteps)

# the incremental loop
tracesToSkip = 20 # warm-up to avoid numerical problems for small evolution step
for i in range (0, tracesToSkip - 1):
    CondAver.addTrace(data[i], traces[i])
for i in range(tracesToSkip - 1, N):
    CondAver.addTrace(data[i], traces[i])

    if (((i + 1) % evolutionStep == 0) or ((i + 1) == N)):

        (avdata, avtraces) = CondAver.getSnapshot()
        
        R2, coefs = lraAES(avdata, avtraces, sBoxOut, basisModelSingleBits)
        print("Result after traces", (i + 1))
        print("LRA")
        print("R2: ", R2)
        R2Peaks = np.max(R2, axis=1) # global maximization
        LraWinningCandidate = np.argmax(R2Peaks)
        LraWinningCandidatePeak = np.max(R2Peaks)
        # LraCorrectCandidateRank = np.count_nonzero(R2Peaks >= R2Peaks[knownKey[SboxNum]])
        # LraCorrectCandidatePeak = R2Peaks[knownKey[SboxNum]]
        print("Winning candidate: 0x%02x, peak magnitude %f", (LraWinningCandidate, LraWinningCandidatePeak))
        print("knowKey", knowKey[SboxNum])
        # print("Correct candidate: 0x%02x, peak magnitude %f, rank %d" ,(knownKey[SboxNum], LraCorrectCandidatePeak, LraCorrectCandidateRank))

        stepCount = int(np.floor(i / np.double(evolutionStep)))
        # keyRankEvolutionLRA[stepCount] = LraCorrectCandidateRank
