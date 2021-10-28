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
plt.plot([first.predict_(np.c_[np.ones(x.shape[0]), x][_]) for _ in range(0, 150)], 'g', label='h(x)')
plt.plot([first.predict(np.c_[np.ones(x.shape[0]), x][_]) for _ in range(0, 150)], 'b', label='thr_h(x)')
plt.plot(y, 'r--', label="Y")
plt.xlabel('Index')
plt.ylabel('Y')
plt.legend()
plt.show()