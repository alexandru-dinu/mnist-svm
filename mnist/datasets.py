from typing import Tuple

import numpy as np
from torchvision import datasets

mnist_mean, mnist_stddev = 0.1307, 0.3081


def get_mnist_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
	print("[+] Fetching data...")

	# 60000 x 28 x 28
	train_loader = datasets.MNIST('../data-mnist', train=True, download=True)
	train_size = train_loader.train_labels.size(0)

	# 10000 x 28 x 28
	test_loader = datasets.MNIST('../data-mnist', train=False, download=True)
	test_size = test_loader.test_labels.size(0)

	train_data = np.zeros(train_loader.train_data.shape)
	train_labels = np.zeros(train_loader.train_labels.shape)
	test_data = np.zeros(test_loader.test_data.shape)
	test_labels = np.zeros(test_loader.test_labels.shape)

	for i in range(train_size):
		train_data[i] = np.array(train_loader.train_data[i].numpy(), dtype=np.float32) / 255.0
		train_labels[i] = train_loader.train_labels[i].numpy()
	# train_data = (train_data - mnist_mean) / mnist_stddev

	for i in range(test_size):
		test_data[i] = np.array(test_loader.test_data[i].numpy(), dtype=np.float32) / 255.0
		test_labels[i] = test_loader.test_labels[i].numpy()
	# test_data = (test_data - mnist_mean) / mnist_stddev

	return (train_data, train_labels), (test_data, test_labels)
