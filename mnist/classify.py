import datetime as dt

from sklearn import svm, metrics

from utils import *


def preview(train: Tuple[np.ndarray, np.ndarray], test: Tuple[np.ndarray, np.ndarray]) -> None:
	x_train, y_train = train
	x_test, y_test = test

	# show some train data
	show_some_digits(x_train, y_train, sample_size=24, normalized=True)
	plt.show()

	# show some test data
	show_some_digits(x_test, y_test, sample_size=24, normalized=True)
	plt.show()


def classify(
		train: Tuple[np.ndarray, np.ndarray], test: Tuple[np.ndarray, np.ndarray],
		classifier: svm.SVC, sizes: Tuple[int, int]) -> None:
	x_train, y_train = train
	x_test, y_test = test

	n_train, n_test = sizes
	idx_train = np.random.choice(x_train.shape[0], n_train)
	idx_test = np.random.choice(x_test.shape[0], n_test)

	print("[+] Training...")
	t = dt.datetime.now()
	classifier.fit(x_train[idx_train].reshape(n_train, -1), y_train[idx_train])
	e = dt.datetime.now() - t
	print(f"Took {e}")

	print("[+] Evaluating...")
	t = dt.datetime.now()
	predicted = classifier.predict(x_test[idx_test].reshape(n_test, -1))
	e = dt.datetime.now() - t
	print(f"Took {e}")

	show_some_digits(x_test[idx_test], predicted, sample_size=24, normalized=True, title_fmt="Predicted {}")
	plt.show()

	cr = metrics.classification_report(y_test[idx_test], predicted)
	print(f"Classification report for classifier {classifier}:\n{cr}\n")

	cm = metrics.confusion_matrix(y_test[idx_test], predicted)
	plot_confusion_matrix(cm, cmap=plt.cm.rainbow)
	plt.show()


def main() -> None:
	mnist_train, mnist_test = get_mnist_data()

	# u, c = np.unique(mnist_train[0].reshape(60000, -1), return_counts=True)
	# plt.bar(u, c, width=0.01)
	# # plt.xticks(np.arange(10))
	# plt.show()
	# exit(0)

	# preview(mnist_train, mnist_test)

	# classifier = svm.SVC(kernel="poly", C=3, gamma=0.01, coef0=0.5)
	classifier = svm.SVC(kernel="rbf", C=2, gamma=0.02)

	classify(mnist_train, mnist_test, classifier, sizes=(3000, 1000))

	pass


if __name__ == "__main__":
	main()
