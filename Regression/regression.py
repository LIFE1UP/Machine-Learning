import numpy as np

class regression:
    def __init__(self, X, Y):
        self.x = np.c_[np.ones(X.shape[0]), X]
        self.y = Y
        self.theta = np.random.rand(self.x.shape[1])
        self.N_1 = 1 / self.x.shape[0]

    def Linear(self, iteration):
        # Loss Function is this: sigma ( learningRate * (y - h(x) / 2)^2 )
        # and to theta to minimize Loss Function value
        for _ in range(0, iteration):
            hx = np.dot(self.x, self.theta)
            self.theta -= 0.0001 * (self.N_1) * np.dot(self.x.T, (hx - self.y))

    def Logistic(self, iteration):
        for _ in range(0, iteration):
            hx = np.dot(self.x, self.theta)
            hx = self.sigmoid(hx)
            self.theta -= 0.0001 * (self.N_1) * np.dot(self.x.T, (hx - self.y))
    def sigmoid(self, z):
        return 1 / (1 + (2.718 ** -z))

    def Locally_Weighted(self, case, tau):
        low = tau ** 2 * -2
        above = (self.x - case) ** 2
        loc_weighting = np.exp(above / low)  # exponential to drop the data that is far from our case, tau is for setting width of our data
        self.x = np.multiply(self.x, loc_weighting)

    def predict(self, type, case):
        if type == 'linear':
            self.Linear(10000)
            return self.thrhold(np.dot(case, self.theta))
        elif type == 'logistic':
            self.Logistic(10000)
            hx = np.dot(case, self.theta)
            if hx > 0.5: return 1
            else: return 0
        else:
            return None



    def thrhold(self, noise):
        self.section = (int(np.argmax(np.unique(self.y))) - int(np.argmin(np.unique(self.y)))) / int(np.unique(self.y).shape[0])
        # Threshold
        for d in range(1, int(np.unique(self.y).shape[0]) + 1):
            if noise > (self.section * d):
                pass
            else:
                return d - 1

    def liner(self):
        return [np.dot(self.x[index], self.theta) for index in range(0, self.x.shape[0])]