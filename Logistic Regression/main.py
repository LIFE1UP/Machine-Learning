import LinearRegression as lnreg
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

# bring datas
iris = datasets.load_iris()
x, y = iris.data, iris.target

# make an instance to make a machine learning
first = lnreg.linear_regression(x, y, 0.0001)

# matplotlib, draw a graph
plt.plot([first.predict_(x[_]) for _ in range(0, 150)], 'b', label='h(x)')
plt.xlabel('Index')
plt.ylabel('Y')
plt.plot(y, 'r', label="Y")
plt.legend()
plt.show()