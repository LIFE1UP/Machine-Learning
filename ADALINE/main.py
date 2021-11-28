import adaline as ada
import multi_classification as mc
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# bring data
x, y = datasets._samples_generator.make_blobs(n_samples=200, n_features=2, centers=2, cluster_std=5, random_state=11)
# make an instance to make a machine learning
x_ = np.c_[np.ones(x.shape[0]), x]

model = ada.perceptron(x, y)
model.learning(cost='MSE', activation='step', iteration=100000, lr=0.0001)
print("")

while 1:
    try:
        index = int(input("index> "))
        case1 = x_[index, :]
        predcition = model.predict(case1, threshold='step')
        print(predcition, y[index])

        fig, ax1 = plt.subplots()
        ax1.scatter(x[:, 0], x[:, 1], c=y)
        ax1.scatter(x[index, 0], x[index, 1], c='b')
        fig.show()
        del fig, ax1

    except:
        continue
    #end
#end
