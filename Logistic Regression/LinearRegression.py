import numpy as np

class linear_regression:
    def __init__(self, X, Y, LearingRate):
        self.x = X
        self.y = Y
        self.lr = LearingRate
        self.theta = np.random.rand(X.shape[1])

    def loss_function(self):
        print(0.5 * np.sum(((np.dot(self.x.T, self.theta) - self.y) ** 2)))

    def gradient(self, iteration):
        for _ in range(0, iteration):
            hx = np.dot(self.x, self.theta.T)
            self.theta -= self.lr * np.dot(self.x.T, (hx - self.y))

    def predict(self, case):
        self.gradient(1000)
        pr = np.dot(case, self.theta)
        if pr >= 1.34:
            return 2
        elif 0.67 < pr < 1.34:
            return 1
        else:
            return 0
    def predict_(self, case):
        self.gradient(1000)
        pr = np.dot(case, self.theta)
        return pr