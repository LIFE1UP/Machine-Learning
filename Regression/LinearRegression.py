import numpy as np

class linear_regression:
    def __init__(self, X, Y):
        self.x = np.c_[np.ones(X.shape[0]), X]
        self.y = Y
        self.theta = np.random.rand(self.x.shape[1])

        # for predict thresholder

        self.persize = (int(np.argmax(np.unique(self.y))) - int(np.argmin(np.unique(self.y)))) / int(np.unique(self.y).shape[0])

    def Linear(self, iteration):
        # Loss Function is this: sigma ( learningRate * (y - h(x) / 2)^2 )
        # and to theta to minimize Loss Function value
        for _ in range(0, iteration):
            hx = np.dot(self.x, self.theta)
            self.theta -= 0.0001 * np.dot(self.x.T, (hx - self.y))

    def Locally_Weighted(self, case, tau):
        low = tau ** 2 * -2
        above = (self.x - case) ** 2
        loc_weighting = np.exp(above / low)  # exponential to drop the data that is far from our case, tau is for setting width of our data
        self.x = np.multiply(self.x, loc_weighting)

    def predict(self, case, type='Linear', thrh = False):
        self.case = case
        if type == 'Linear':
            self.Linear(1000)
        elif type == "Locally_Weighted":
            self.Locally_Weighted(case, 2)
            self.Linear(10000)
        else:
            print("type your loss function")

        if thrh == True:
            self.thrhold(np.dot(case, self.theta))
        else:
            return np.dot(case, self.theta)

    def thrhold(self, noise):
        # Threshold
        for d in range(1, int(np.unique(self.y).shape[0])):
            if noise > (self.persize * d):
                pass
            else:
                return d - 1
        return int(np.argmax(np.unique(self.y)))

    def inspect(self, command):
        if command == "Loss Function":
            print(0.5 * np.sum(((np.dot(self.x.T, self.theta) - self.y) ** 2)))
        elif command == "Weight":
            print(self.theta)
        elif command == "h(x)":
            self.pr = np.dot(self.case.T, self.theta)
            print(self.pr)
        else:
            print("Parameter Error")