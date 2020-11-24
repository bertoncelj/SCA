import numpy as np
import matplotlib.pyplot as plt
from  numpy import linalg as LA
import statsmodels.api as sm

from condaveraes import * # incremental conditional averaging
import functools
import operator


SboxNum = 0
TRACES_NUMBER = 200
TRACE_LENGTH = 1000 ## number of samples
TRACE_STARTING_SAMPLE = 800
offset = 0
KNOW_KEY = b'\x2b\x7e\x15\x16\x28\xae\xd2\xa6\xab\xf7\x15\x88\x09\xcf\x4f\x3c'

# for faster calculation 
sbox = np.load('data/aessbox.npy')

traceRange = range(0, TRACES_NUMBER)
sampleRange = (TRACE_STARTING_SAMPLE, TRACE_STARTING_SAMPLE + TRACE_LENGTH) 
# load traces, data
npzfile = np.load('traces/swaes_atmega_power.trs.npz')
data = npzfile['data'][:, SboxNum]
loadTraces = npzfile['traces'][traceRange, TRACE_LENGTH]
traces = npzfile['traces'][offset:offset + TRACES_NUMBER,sampleRange[0]:sampleRange[1]]

print(traces)

foldl = lambda func, acc, xs: functools.reduce(func, xs, acc)

def mean(vecArr):
    sumArr = foldl(operator.sub, 0, vecArr)
    
    return sumArr/vecArr.size
def lraAES(data, traces, sBox):
    yi = mean(traces[0])

    
    SStot = np.sum((traces - np.mean(traces, 0)) ** 2, 0)

    ### 2. The main attack loop

    # preallocate arrays
    SSreg = np.empty((64, traceLength)) # Sum of Squares due to regression
    E = np.empty(numTraces)              # expected values

    allCoefs = [] # placeholder for regression coefficient



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


infoNpzFile(npzfile)
# printTrace(npzfile['traces'])


#build prediction
# keyByte = np.uint8(KNOW_KEY[0])
# sBoxOut = sbox[loadData ^ keyByte]
# print("loadData: ", loadData)
# print("sBoxOut: ", sBoxOut)


(numTraces, traceLength) = traces.shape
CondAver = ConditionalAveragerAesSbox(256, traceLength)

for i in range (0, 200): #200 trace attack
    CondAver.addTrace(data[i], traces[i])


(avdata, avtraces) = CondAver.getSnapshot()
lraAES(avdata, avtraces, data)



# print("HHHH", CondAver)

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

