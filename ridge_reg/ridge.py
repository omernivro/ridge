import numpy as np

class ridge(object):
    trials = 10
    regularizer = []
    """docstring for ClassName"""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def mse(self, y_true, y_hat):
        self.err = sum(square(np.asarray(y_true - y_hat))) / (y_true.shape[0])

        return(self.err)


    def choose_best_comb(self):
        pass

    def store_results(self):
        pass


def main():
    pass

if __name__ == '__main__':
    main()
