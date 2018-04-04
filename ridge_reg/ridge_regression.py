
# coding: utf-8

import numpy as np
from numpy.linalg import inv
import decimal


# source: Kernel Methods for Pattern Analysis by J.shaw Taylor

'''embed the data into a space where the patterns can be discovered
as linear relatio'''
# a valid kernel is one which is positive semi definite, i.e all eigen
# values are ≥ 0, meaning the kernel matrix scales space either
# positively or by zero.

'''Use primal ridge when there are more training examples than dimensions
use dual ridge when there are more dimensions than training examples'''



def ridge_kernel_cv(x, y, trails, tr_perc, val_perc, typ, regu, sigma=None, cv=5):

    cv_errs = []
    err_w_params = []


    for trial in xrange(trails):

        tr_val_idx, te_idx = split_tr_te(x.shape[0], tr_perc=0.8)

        x_tr_val, y_tr_val, x_te, y_te = x[tr_val_idx], y[tr_val_idx], x[te_idx], y[te_idx]

        for comb in gen_regu_sigma_combo:

            for iter_num in xrange(cv):

                regulizer = comb[0]
                sig = ifelse(len(comb) == 2, comb[1], None)

                x_tr, y_tr, x_val, y_val, tr_idx, val_idx = cv_split(
                    cv, iter_num, x_tr_val, y_tr_val, tr_val_idx)

                ker_tr, ker_val_tr, ker_val, ker_te_tr, ker = ker_mat(
                    x, tr_idx, val_idx, te_idx, typ, sigma=sig)

                alp_dual, alp_val, hat_tr_val, hat_te, hat_val = dual_ridge(
                    regulizer, ker_tr, ker_val, ker_val_tr, ker_te_tr, y_tr, y_tr_val)

                val_err, tr_err, te_err = mean_error(
                    y_tr_val, y_val, y_te, hat_tr_val, hat_val, hat_te)

                cv_errs.append([tr_err, val_err, te_err])

            err_w_params.append([comb, mean(cv_errs)])


    return(err_w_params)


def main():
    pass


if __name__ == '__main__':
    main()
