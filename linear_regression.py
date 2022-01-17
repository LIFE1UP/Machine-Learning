from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

# set data pipline and hyper-parameters
data = load_iris()
shuffledIndices = np.random.permutation(len(data.target))
data.data, data.target = data.data[shuffledIndices], data.target[shuffledIndices]
trainX, testX = data.data[:100,1:3], data.data[100:,1:3]
trainY, testY = data.target[:100], data.target[100:]
weight = np.random.rand(trainX.shape[1])
lr = 0.001
iterations = 10000

# define activation functions
nothing = lambda x: x
sigmoid = lambda x: 1 / (1 + 2.7182**(-x))
bipolar = lambda xSet: np.array([-1 if x < 0 else 1 for x in xSet])

# activation fn dictionary
actDict = {"nothing":nothing,"sigmoid":sigmoid,"bipolar":bipolar}

# define gradient descent rule
def gradient_descent_rule(name):
    global trainX, trainY, weight, lr
    
    activHX = actDict[name](np.dot(trainX, weight))
    weight -= (lr / trainX.shape[0]) * np.dot(trainX.T, (activHX - trainY))
# def

for epoch in range(iterations):
    gradient_descent_rule("nothing")  # don't touch these
# for

# define predict
def predict(case):
    global weight
    
    if type(case) == list:
        case = np.array(case)
    
    return round(np.dot(case, weight))
# def

index = random.randrange(len(testY))
prediction = predict(testX[index,:])

# cost funtion
def mean_squared_error():
    global testX, testY
    
    predictY = np.dot(testX, weight)
    loss = np.sum(np.dot(testX.T, (predictY - testY)**2))
    
    print(f"cost: {loss}")
# MSE

def mean_absolute_error():
    global testX, testY
    
    predictY = np.dot(testX, weight)
    loss = np.sum(np.dot(testX.T, abs(predictY - testY)))
    
    print(f"cost: {loss}")
# MAE

mean_squared_error()
mean_absolute_error()

# <visualization>
# making decision boundaries
baseX = np.linspace(trainX[:,0].min(), trainX[:,0].max(), 2)
boundary1, boundary2 = list(), list()
i, counter = 0, 0

for n in baseX:
    while i < trainX[:,1].max():
        if predict([n,i]) == 1. and counter == 0:
            boundary1.append([n,i])
            counter += 1
        
        if predict([n,i]) == 2. and counter == 1:
            boundary2.append([n,i])
            break
        i += 0.1
    # while
    i = 0
    counter = 0
# for

boundary1 = np.array(boundary1)
boundary2 = np.array(boundary2)

# matplotlib
plt.figure(figsize=(5,5))
plt.scatter(x=trainX[:,0], y=trainX[:,1], s=50, alpha=1, c=trainY)
plt.scatter(x=testX[index,0], y=testX[index,1], c='r', s=200, marker='+')
plt.plot(boundary1[:,0], boundary1[:,1], c='black')
plt.plot(boundary2[:,0], boundary2[:,1], c='black')
plt.xlim(trainX[:,0].min() - 0.5, trainX[:,0].max() + 0.5)
plt.grid()
plt.xlabel(f"Prediction: {prediction:.0f}, Actual Target: {testY[index]}")
plt.ylabel("Yellow = 2, Green = 1, Violet = 0")
plt.title("load_iris")
plt.show()
