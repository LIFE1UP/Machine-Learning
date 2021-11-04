import regression as reg
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np


def command(X, Y, case, type, option):
    reg1 = reg.regression(X, Y)

    # activate option
    if option == "locally_weighted":
        lnreg1.Locally_Weighted(case, 2)
    else:
        print("using default option")
        pass

    # which type?
    if type == 'linear':
        print("my estimation is", reg1.predict('linear', case))
    elif type == 'logistic':
        print("my estimation is", reg1.predict('logistic', case))
    else:
        print("no specified type")

    if switcher == True:
        print("and actual y is", y[int(index)], "\nfinal theta is", reg1.theta)
    else:
        pass

    return reg1

# bring data
x, y = datasets._samples_generator.make_blobs(n_samples=200, n_features=2, centers=2, cluster_std=5, random_state=11)
# make an instance to make a machine learning
x_ = np.c_[np.ones(x.shape[0]), x]

while 1:
    switcher = False
    print("case> ", end='')
    case_ = input()
    if case_ == 'exit': break
    if case_ == "uad":
        print("index> ", end='')
        index = input()
        case_ = x_[int(index), :]
        switcher = True
    else:
        try:
            case_ = case_.split(" ")
            case_ = np.array([int(case_[ind]) for ind in range(0, len(case_))])
        except:
            break

    print("type> ", end='')
    type_ = input()
    if type_ == 'exit': break

    print("option> ", end='')
    option_ = input()
    if option_ == 'exit': break

    my_case = command(x, y, case_, type_, option_)

    plt.title('red = 0 green = 1')
    color = ['red' if l == 0 else 'green' for l in y]
    plt.scatter(x[:, 0], x[:, 1], color=color, s=[10])
    try: plt.scatter(x[int(index), 0], x[int(index), 1], s=[30], color='blue')
    except: plt.scatter(case_[1], case_[2], s=[30], color='blue')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

# matplotlib, draw a graph
#plt.plot([first.predict(np.c_[np.ones(x.shape[0]), x][_], "MSE") for _ in range(0, 150)], 'b', label='h(x)')