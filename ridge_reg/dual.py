import numpy as np
from numpy.linalg import inv
import itertools


class dual(ridge):
    """docstring for ClassName"""
    kernel_type = 'lin'
    sigma = None

    def __init__(self, arg):
        super(ridge, self).__init__()
        self.arg = arg

    def create_params_combo(self, regularizer):
        self.combo = list(itertools.product(regu, sigma))
        return(self.combo)

    def gaus_kernel(self, X1, sigma):
        '''page 77 J.shaw taylor
        also called RBF kernel, it corresponds
        to applying a gausian with mean at point z and get the prob. that
        point x came from that that gausian dist.. So a prediction at a
        new point can be viewed as a weighted combination of my
        probability to belong to any one of the gausians'''
        nrow, ncol = np.shape(X1)
        K = np.matmul(X1, np.transpose(X1)) / (sigma ** 2)
        d = np.diag(K)
        K1 = K - (d.reshape(-1, 1) / (2 * (sigma ** 2)))
        K2 = K1 - (d.reshape(1, -1) / (2 * (sigma ** 2)))
        K3 = np.exp(K2)
        return K3

    def calc_kernel_mat(self, kernel_type, sigma=None):
        if (typ == 'lin'):
            ''' corresponds to regular linear regression'''
            ker = np.asmatrix(x.dot(x.transpose()))

        if (typ == 'quad'):
            '''corresponds as one example to
            feature map (x1^2, x2^2, √2*x1*x2) '''
            ker = np.asmatrix(
                np.square(np.asarray(self.x.dot(self.x.transpose()))))

        if (typ == 'gaus'):
            '''Infinite dimensional kernel'''
            ker = gaus_kernel(self.x, sigma)

        return(ker)

    def predict(self, tr_kernel, tr_y, regu):
        self.alpha = (
            inv(tr_kernel + (regu) * (np.eye(tr_kernel.shape[0])))).dot(tr_y)

        y_hat = self.tr_kernel.dot(self.alpha)


def main():
    pass


if __name__ == '__main__':
    main()

