import adaline as ada

class multi_classification:
    def __init__(self, X, Y, class_numbers):
        self.cn = class_numbers - 1
        self.perceptrons = [ada.perceptron(X, Y[:, i]) for i in range(self.cn - 1)]

    # end
    def learning(self, cost, activation, iteration, lr=0.0001):
        for i in range(self.cn - 1):
            self.perceptrons[i].learning(cost, activation, iteration, lr)
        # end

    # end
    def predict(self, case, threshold):
        prs = []
        for i in range(self.cn - 1):
            prs[i] = self.percpetrons[i].predict(case, threshold)
        # end
        return prs

    # end
    def softmax(self, case):
        prs = []
        sums = 0
        for i in range(self.cn):
            prs[i] = 2.718 ** (self.percpetrons[i].predict(case, 'none'))
            sums += prs[i]
        # end
        for i in range(self.cn):
            prs[i] = prs[i] / sums
        del sums
        return prs
    # end
#end
