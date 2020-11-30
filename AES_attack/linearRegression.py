import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



dataset = pd.read_csv('Salary_Data.csv')
print(dataset.head())



X = dataset.iloc[:,:-1].values #indpnedet variable array
Y = dataset.iloc[:,:1].values  #dependent variable array

print("X len: ",  len(X))
print("Y len: ",  len(Y))


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)


print("X train: ",  X_train)
print("Y train: ",  Y_train)
print("X test: ",  X_test)
print("Y test: ",  Y_test)


from sklearn.linear_model import LinearRegression
reg= LinearRegression()
reg.fit(X_train, Y_train)

y_pred = reg.predict(X_test)
print("y_pred: ",y_pred)


#plot for the TRAIN
 
plt.scatter(X_train, Y_train, color='red') # plotting the observation line
 
plt.plot(X_train, reg.predict(X_train), color='blue') # plotting the regression line
 
plt.title("Salary vs Experience (Training set)") # stating the title of the graph
 
plt.xlabel("Years of experience") # adding the name of x-axis
plt.ylabel("Salaries") # adding the name of y-axis
plt.show() # specifies end of graph
