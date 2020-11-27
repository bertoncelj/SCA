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

# = 255 for faster calculation 
sbox = np.load('data/aessbox.npy')

traceRange = range(0, TRACES_NUMBER)
sampleRange = (TRACE_STARTING_SAMPLE, TRACE_STARTING_SAMPLE + TRACE_LENGTH)
# load traces, data
npzfile = np.load('traces/swaes_atmega_power.trs.npz')
data = npzfile['data'][:, SboxNum]
traces = npzfile['traces'][offset:offset + TRACES_NUMBER,sampleRange[0]:sampleRange[1]]

print(traces)
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

def getBeta(appliedModelData, trace):
    # Equation:
    # Beta = (X.T * X)^(-1) * (X.T * Y)
    Y = trace
    X = appliedModelData

    X = np.asarray(appliedModelData)
    leva = np.dot(X.T, X)
    leva = np.linalg.inv(leva)
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
    for key in range(0, 3):
        print(key)
        for positionTraceSample in range(0, TRACE_LENGTH):
            # print("trace sample : ", positionTraceSample)
            # print(traces[:, positionTraceSample])
            traceToAtt = traces[:, positionTraceSample]

            appliedModelData = list(map(basisModelSingleBits, keyDataPredictions[key]))
            # print("beta in : ", keyDataPredictions[0])
            # print("beta in : ", traceToAtt)
            beta = getBeta(appliedModelData, traceToAtt)
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
rez = attack(avdata, avtraces)
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
# print(R2rtn)
# R2Peaks = np.max(R2rtn, axis=1) # global maximization
# LraWinningCandidate = np.argmax(R2Peaks)
# LraWinningCandidatePeak = np.max(R2Peaks)
# print(LraWinningCandidatePeak) 
# print("_"*100)
#         # LraCorrectCandidateRank = np.count_nonzero(R2Peaks >= R2Peaks[knownKey[SboxNum]])
# # print("HHHH", CondAver)

# X = list(map(leakageModel2, sBoxOut))
# A = sm.add_constant(X, prepend=False)

# infoMatrixDemensions(traces)
# infoMatrixDemensions(A)
# infoMatrixDemensions(X)
# print(A)
# results = sm.OLS(traces, X).fit() # the OLS itself
# betaOLS = results.params 
# print(results.summary())

'''
dataset = pd.read_csv('Salary_Data.csv')
print(dataset.head())
X = dataset["Salary"]
print("data: ", X[:])

# X = dataset.iloc[:,:-1].values #indpnedet variable array
Y = dataset.iloc[:,:1].values  #dependent variable array
print("X: ", X)
print("Y: ", Y)
print("X len: ",  len(X))
print("Y len: ",  len(Y))
X_new = []
for x in X:
    X_new.append(x)
print(X_new)

X = np.array([X_new])
X = X.T
Y = np.array(Y)

print("X: ", X)
print("Y: ", Y)
Rsquare(Y, X)

dummy = np.ones(len(X))
dummy = dummy.astype('float64')
print(dummy.dtype)
print(X.dtype)

print("X: ", X.ravel())
print("dummy: ", dummy)


# X= np.hstack((np.atleast_2d(dummy).T, X))

X = sm.add_constant(X, prepend=False) # add constant coefficient (trailing column of ones)
M = X
P = np.dot(np.linalg.inv(np.dot(M.T, M)), M.T)

beta = np.dot(P, Y)
print("FINAL BETA: ", beta)


print("-"*30, "OLS")

print("X: ", X)
print("Y: ", Y)

results = sm.OLS(Y, X).fit() # the OLS itself
betaOLS = results.params 
print(results.summary())

print("beta OLS: ", betaOLS)

# X= np.hstack((np.atleast_2d(dummy).T, X))
print(X)
xsi = np.dot(X.T, X)
print(xsi)
leva = np.linalg.inv(xsi)
print(leva)
desna = np.dot(X.T, Y)
print(desna)
beta = np.dot(leva, desna)

print("beta:", beta)
x_mean = np.mean(X)
y_mean = np.mean(Y)

SSxy=0
for i in range(0,len(X)):
    SSxy = SSxy + (Y[i]*X[i]-len(X)*x_mean*y_mean)
print(SSxy)
SSxx=0
for i in range(0,len(X)):
    SSxx = SSxx + ((X[i]**2)-len(X)*x_mean**2)

print(SSxx)

B1 = SSxy/SSxx
B0 = y_mean - B1*x_mean
print("B1:", B1)
print("B0:", B0)
print("betaOLS0:", betaOLS[1])
print("betaOLS1:", betaOLS[0])


y_rez1 = []
for i in X[:,0]:
    print("i:", i)
    y_rez1.append(B1*i+B0)

y_rez2 = []
for i in range(0,len(X[:,0])):
    y_rez2.append(betaOLS[0]*i+betaOLS[1])

print("X len:", len(X[:,0]))
print(len(y_rez1))
print(len(y_rez2))
print("X[:,0]", X[:,0])
print("y_rez1:", y_rez1)
plt.scatter(X[:,0], Y, color='red') # plotting the observation line
plt.scatter(X[:,0], y_rez1[:,0])
plt.scatter(X[:,0], y_rez2, color='green')


plt.xlabel("Years of experience") # adding the name of x-axis
plt.ylabel("Salaries") # adding the name of y-axis
plt.show() # specifies end of graph
'''

