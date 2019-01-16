"""
MNIST classification using Support Vector algorithm with RBF kernel
all parameters are optimized by grid search cross validation
"""

import datetime as dt

from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV

from utils import *


def generate_parameters():
	gamma_range = np.outer(np.logspace(-3, 0, 4), np.array([1, 2, 5]))
	gamma_range = gamma_range.flatten()

	C_range = np.outer(np.logspace(-1, 1, 3), np.array([1, 2, 5]))
	C_range = C_range.flatten()

	parameters = {'kernel': ['rbf'], 'C': C_range, 'gamma': gamma_range}

	return parameters


def grid_search(parameters, x_train, y_train):
	svm_clsf = svm.SVC()
	grid_clsf = GridSearchCV(estimator=svm_clsf, param_grid=parameters, n_jobs=-1, verbose=2)

	t = dt.datetime.now()
	print("[+] Grid search...")

	grid_clsf.fit(x_train, y_train)

	print(f"Took {dt.datetime.now() - t}")

	sorted(grid_clsf.cv_results_.keys())

	# eval
	scores = grid_clsf.cv_results_['mean_test_score'].reshape(
		len(parameters['C']), len(parameters['gamma'])
	)
	plot_param_space_heatmap(scores, parameters['C'], parameters['gamma'])

	return grid_clsf.best_estimator_, grid_clsf.best_params_


def evaluate(classifier, x_test, y_test) -> None:
	expected = y_test
	predicted = classifier.predict(x_test)

	show_some_digits(x_test, predicted, sample_size=24, normalized=True, title_fmt="Predicted {}")
	plt.show()

	cr = metrics.classification_report(expected, predicted)
	print(f"Classification report for classifier {classifier}:\n{cr}\n")

	cm = metrics.confusion_matrix(expected, predicted)
	plot_confusion_matrix(cm)
	plt.show()

	acc = metrics.accuracy_score(expected, predicted)
	print(f"Accuracy = {acc}")


def main() -> None:
	(x_train, y_train), (x_test, y_test) = get_mnist_data()

	params = generate_parameters()

	n_train, n_test = 1000, 50
	idx_train = np.random.choice(x_train.shape[0], n_train)
	idx_test = np.random.choice(x_test.shape[0], n_test)

	best_svm, _ = grid_search(params, x_train[idx_train].reshape(n_train, -1), y_train[idx_train])

	evaluate(best_svm, x_test[idx_test].reshape(n_test, -1), y_test[idx_test])


if __name__ == "__main__":
	main()
