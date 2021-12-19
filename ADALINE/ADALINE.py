import numpy as np

def Counter_intf(itearation, curr):
    if curr % (itearation / 100) == 0:
        print("#", end="")


# Class of Model
class perceptron:
    def __init__(self, X, Y):
        self.x = X
        self.y = Y
        self.theta = np.random.rand(self.x.shape[1])
        self.N_1 = 1 / self.x.shape[0]


# Activation/Threshold f(x)
    def act_choice(self, inpt, name):
        if name == 'sigmoid':
            return 1.0 / (1.0 + (2.718 ** -z))

        return inpt


# Gradient Descent Rule: Choose your cost function!
    def GDR_MSE(self, activation='noneactivation'):
        hx = np.dot(self.x, self.theta)
        hx = self.act_choice(hx, activation)
        self.theta -= (self.lr * self.N_1) * np.dot(self.x.T, (hx - self.y))
        del hx

    def GDR_LogLike(self, activation='sigmoid'):
        hx = np.dot(self.x, self.theta)
        hx = self.act_choice(hx, activation)
        self.theta += self.lr * (self.N_1) * np.dot(self.x.T, (self.y - hx))
        del hx

    def cost_choice(self, name_c, name_a):
        if name_c == 'LogLike':
            self.GDR_LogLike(name_a)

        return self.GDR_MSE(name_a)


# Options
    def localeight(self, case, tau):
        comp = ((self.x - case) ** 2) / (tau ** 2 * -2)
        localized = np.exp(comp)
        self.x = np.multiply(self.x, localized)


# Predict
    def learning(self, cost, activation, iteration, lr=0.0001):
        self.lr = lr
        for iters in range(iteration):
            self.cost_choice(cost, activation)
            Counter_intf(iteration, iters)
        del self.lr

    def predict(self, case, threshold):
        predicted_y = np.dot(case, self.theta)
        return self.act_choice(inpt=predicted_y, name=threshold)
