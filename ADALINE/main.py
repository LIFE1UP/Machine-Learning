import adaline as ada
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# bring data
x, y = datasets._samples_generator.make_blobs(n_samples=200, n_features=2, centers=2, cluster_std=5, random_state=11)
x = np.c_[np.ones(x.shape[0]), x]

model = ada.perceptron(x, y)
model.learning(cost='MSE', activation='sigmoid', iteration=100000, lr=0.0001)
print("")

while 1:
    try:
        index = int(input("index> "))
        case = x[index, :]

        print(f"Prediction: {round(model.predict(case, threshold='sigmoid'))} Actual Target: {y[index]}")

        fig, ax1 = plt.subplots()
        ax1.scatter(x[:, 1], x[:, 2], c=y)
        ax1.scatter(x[index, 1], x[index, 2], c='b')
        fig.show()
        del fig, ax1

    except:
        continue
