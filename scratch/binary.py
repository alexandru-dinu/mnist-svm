import random
import numpy as np

import svm
import svmutil

data_path = "../data/mushrooms.data"

"""
Examples of options: -s 0 -c 10 -t 1 -g 1 -r 1 -d 3 
Classify a binary data with polynomial kernel (u'v+1)^3 and C = 10

 
options:
-s svm_type : set type of SVM (default 0)
	0 -- C-SVC
	1 -- nu-SVC
	2 -- one-class SVM
	3 -- epsilon-SVR
	4 -- nu-SVR
-t kernel_type : set type of kernel function (default 2)
	0 -- linear: u'*v
	1 -- polynomial: (gamma*u'*v + coef0)^degree
	2 -- radial basis function: exp(-gamma*|u-v|^2)
	3 -- sigmoid: tanh(gamma*u'*v + coef0)
-d degree : set degree in kernel function (default 3)
-g gamma : set gamma in kernel function (default 1/num_features)
-r coef0 : set coef0 in kernel function (default 0)
-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
-m cachesize : set cache memory size in MB (default 100)
-e epsilon : set tolerance of termination criterion (default 0.001)
-h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
-b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
-wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)

The k in the -g option means the number of attributes in the input data.
"""


def gaussian_rbf(ys_test, xs_test, ys_train, xs_train):
    """
    exp(-gamma*|u-v|^2)
    vary gamma = 1/2s^2
    """
    results = {}
    num_features = len(xs_test[0])

    # gammas = np.logspace(-4, 0, num=20)
    gammas = [1.0 / num_features]

    for i, g in enumerate(gammas):
        svm_params = '-t 2 -g ' + str(round(g, 5))
        print("{}".format(i), svm_params, "-" * 80)

        m = svmutil.svm_train(ys_train, xs_train, svm_params)
        p_label, p_acc, p_val = svmutil.svm_predict(ys_test, xs_test, m)

        results[g] = p_acc

    for g, acc in results.items():
        print("{} gives {}".format(g, acc))


def polinomial(ys_test, xs_test, ys_train, xs_train):
    """
    (gamma*u'*v + coef0)^degree
    vary gamma, coef0, degree
    """
    results = {}

    num_features = len(xs_test[0])

    gammas = [0.001, 0.005, 0.01, 1 / num_features, 0.1]
    thetas = [-5, -2, 0, 2, 5]
    degrees = [1, 2, 3, 4]

    i = 0

    for d in degrees:
        for g in gammas:
            for t in thetas:
                svm_params = '-h 0 -t 1 -d ' + str(d) + ' -g ' + str(g) + ' -r ' + str(t)
                info = "{}".format(i) + svm_params

                print(info)

                m = svmutil.svm_train(ys_train, xs_train, svm_params)
                p_label, p_acc, p_val = svmutil.svm_predict(ys_test, xs_test, m)

                results[(d, g, t)] = p_acc

                i += 1
                print("-" * 80)

    for params, acc in results.items():
        print("{} gives {}".format(params, acc))


def liniar(ys_test, xs_test, ys_train, xs_train):
    """
    u'*v
    nothing to vary
    """
    svm_params = '-t 0'
    m = svmutil.svm_train(ys_train, xs_train, svm_params)
    p_label, p_acc, p_val = svmutil.svm_predict(ys_test, xs_test, m)


def main():
    ys, xs = svmutil.svm_read_problem(data_path)
    data = list(zip(ys, xs))
    random.shuffle(data)

    split = 1500
    # split = 2000
    ys_test, xs_test = [v[0] for v in data[:split]], [v[1] for v in data[:split]]
    ys_train, xs_train = [v[0] for v in data[split:]], [v[1] for v in data[split:]]

    tests = [
        gaussian_rbf, polinomial, liniar
    ]

    tests[1](ys_test, xs_test, ys_train, xs_train)


if __name__ == '__main__':
    main()
