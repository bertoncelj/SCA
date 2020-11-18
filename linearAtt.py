import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  numpy import linalg as LA
import statsmodels.api as sm


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
