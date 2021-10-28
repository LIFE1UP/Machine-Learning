import numpy as np

class linear_regression:
    def __init__(self, X, Y, LearingRate):
        self.x = np.c_[np.ones(X.shape[0]), X]
        self.y = Y
        self.lr = LearingRate
        self.theta = np.random.rand(self.x.shape[1])

        # for predict thresholder

        self.persize = (int(np.argmax(np.unique(self.y))) - int(np.argmin(np.unique(self.y)))) / int(np.unique(self.y).shape[0])

    def gradient(self, iteration):
        for _ in range(0, iteration):
            hx = np.dot(self.x, self.theta.T)
            self.theta -= self.lr * np.dot(self.x.T, (hx - self.y))

    def predict(self, case):
        self.gradient(1000)
        self.pr = np.dot(case, self.theta)

        for d in range(1, int(np.unique(self.y).shape[0])):
            if self.pr > (self.persize * d):
                pass
            else:
                return d - 1
        return int(np.argmax(np.unique(self.y)))

    def predict_(self, case):
        self.gradient(1000)
        return np.dot(case, self.theta)

    def inspect(self, command):
        if command == "Loss Function":
            print(0.5 * np.sum(((np.dot(self.x.T, self.theta) - self.y) ** 2)))
        elif command == "Weight":
            print(self.theta)
        elif command == "h(x)":
            print(self.pr)
        else:
            print("Parameter Error")