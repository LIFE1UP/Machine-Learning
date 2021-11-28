import numpy as np

def Counter_intf(itearation, curr):
    if curr % (itearation / 100) == 0:
        print("#", end="")
    else:
        pass
#end

class perceptron:
    def __init__(self, X, Y):
        # creating field for machine learning
        self.x = np.c_[np.ones(X.shape[0]), X]
        self.y = Y
        self.theta = np.random.rand(self.x.shape[1])
        self.N_1 = 1 / self.x.shape[0]
    #end

# activation/threshold functions
    def none(self, z):
        return z
    #end

    def step(self, z):
        return 0 if z < 0.5 else 1
    #end

    def sigmoid(self, z):
        return 1.0 / (1.0 + (2.718 ** -z))
    #end

    def act_choice(self, inpt, name):
        if name == 'none':
            return self.none(inpt)
        elif name == 'step':
            return self.step(inpt)
        elif name == 'sigmoid':
            return self.sigmoid(inpt)
        else:
            print("set activation function / function function")
    #end

# Gradient Descent Rule... Choose your cost function!
    def GDR_MSE(self, activation='noactivation'):
        # Calulate gradient and update theta
        hx = np.dot(self.x, self.theta)
        self.theta -= (self.lr * self.N_1) * np.dot(self.x.T, (hx - self.y))
        del hx
    #end

    def GDR_LogLike(self, activation='sigmoid'):
        hx = np.dot(self.x, self.theta)
        hx = self.act_choice(hx, activation)  # activation function (sigmoid)
        self.theta += self.lr * (self.N_1) * np.dot(self.x.T, (self.y - hx))
        del hx
    #end

    def GDR_Cross(self,  activation='sigmoid'):
        hx = np.dot(self.x, self.theta)
        hx = self.act_choice(hx, activation)
        self.theta += self.lr * (self.N_1) * np.dot(self.x.T, (self.y - hx))
        del hx
    #end

    def cost_choice(self, name_c, name_a):
        if name_c == 'MSE':
            self.GDR_MSE(name_a)
        elif name_c == 'LogLike':
            self.GDR_LogLike(name_c)
        elif name_c == 'Cross-Entropy':
            self.GDR_Cross(name_c)
    #end

# options
    def localeight(self, case, tau):
        # With normalized model, we can ignore data might be useless for classification
        low = tau ** 2 * -2
        above = (self.x - case) ** 2
        localized = np.exp(above / low)  # exponential to drop the data that is far from our case, tau is for setting width of our data
        self.x = np.multiply(self.x, localized)

# predict
    def learning(self, cost, activation, iteration, lr=0.0001):
        self.lr = lr
        for iters in range(iteration):
            self.cost_choice(cost, activation)
            Counter_intf(iteration, iters)
        #end

    def predict(self, case, threshold):
        predicted_y = np.dot(case, self.theta)
        return self.act_choice(inpt=predicted_y, name=threshold)
    #end
#end
