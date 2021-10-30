import LinearRegression as lnreg
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def add_bias(x):
    fixed_x = np.c_[np.ones(x.shape[0]), x]
    return fixed_x

# bring data
iris = datasets.load_iris()
x, y = iris.data, iris.target
# make an instance to make a machine learning
reg1 = lnreg.linear_regression(x, y)
x_ = add_bias(x)

# casing our things

# matplotlib, draw a graph
#plt.plot([first.predict(np.c_[np.ones(x.shape[0]), x][_], "MSE") for _ in range(0, 150)], 'b', label='h(x)')
plt.plot(x_[50, 4], [reg1.predict(x_[50], 'Locally_Weighted')], 'g^', label='loc_h(x)')
reg1.inspect(command="h(x)")
plt.plot(x_[:, 4], y, 'ro', label="Y")
plt.xlabel('Index')
plt.ylabel('Y')
plt.legend()
plt.show()