import random
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import svmutil
from mwv_svm import mwv_svm
from wta_svm import wta_svm

train_path = "../data/pendigits/pendigits"
test_path = "../data/pendigits/pendigits.t"

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

"""
RBF Gaussian / Polinomial
"""


def read_dataset(path, shuffle=False):
	ys, xs = svmutil.svm_read_problem(path)

	if not shuffle:
		return ys, xs

	data = list(zip(ys, xs))
	random.shuffle(data)

	ys_shuf, xs_shuf = [], []

	for v in data:
		ys_shuf.append(v[0])
		xs_shuf.append(v[1])

	return ys_shuf, xs_shuf


def get_data():
	ys_train, xs_train = read_dataset(train_path, shuffle=True)
	ys_test, xs_test = read_dataset(test_path, shuffle=True)

	return (ys_train, xs_train), (ys_test, xs_test)


def preview(train, test):
	(ys_train, xs_train), (ys_test, xs_test) = train, test

	x_counts = defaultdict(lambda: 0)
	y_counts = defaultdict(lambda: 0)

	xs = xs_train + xs_test
	ys = ys_train + ys_test
	for i in range(len(xs)):
		for k in xs[i].keys():
			x_counts[k] += 1
		y_counts[ys[i]] += 1

	fig, ax = plt.subplots(figsize=(10, 10))
	ax.bar(y_counts.keys(), y_counts.values(), 0.5, color='g')
	plt.xticks(np.arange(10))
	# plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
	plt.show()
	pass


def main():
	(ys_train, xs_train), (ys_test, xs_test) = get_data()

	# preview((ys_train, xs_train), (ys_test, xs_test))
	# exit(0)

	k2i = {
		'rbf' : 2,
		'poly': 1
	}

	methods = {
		"wta": wta_svm,
		"mwv": mwv_svm
	}

	methods[sys.argv[1]](ys_test, xs_test, ys_train, xs_train, ktype=k2i[sys.argv[2]], gamma=float(sys.argv[3]))


if __name__ == '__main__':
	main()
