import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def show_some_digits(images, targets, sample_size=24, normalized=False, title_fmt="Digit {}"):
	rand_idx = np.random.choice(images.shape[0], sample_size)
	images_and_labels = list(zip(images[rand_idx], targets[rand_idx]))

	img = plt.figure(1, figsize=(15, 12), dpi=160)
	for index, (image, label) in enumerate(images_and_labels):
		plt.subplot(np.ceil(sample_size / 6.0), 6, index + 1)
		plt.axis('off')

		if normalized:
			image = np.array(255 * image).astype(np.uint8)

		plt.imshow(image.reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
		plt.title(title_fmt.format(int(label)))


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
	plt.figure(1, figsize=(8, 8), dpi=80)
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.xticks(np.arange(10))
	plt.yticks(np.arange(10))
	plt.xlabel('Predicted label')
	plt.ylabel('True label')
	plt.title(title)
	plt.colorbar()
	plt.tight_layout()


class MidpointNormalize(Normalize):
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y))


def plot_param_space_heatmap(scores, C_range, gamma_range):
	"""
	Draw heatmap of the validation accuracy as a function of gamma and C

	Parameters
	----------
	scores - 2D numpy array with accuracies

	# The score are encoded as colors with the hot colormap which varies from dark
	# red to bright yellow. As the most interesting scores are all located in the
	# 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
	# as to make it easier to visualize the small variations of score values in the
	# interesting range while not brutally collapsing all the low score values to
	# the same color.
	"""

	plt.figure(figsize=(8, 6))
	plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
	plt.imshow(scores, interpolation='nearest', cmap=plt.cm.jet,
			   norm=MidpointNormalize(vmin=0.5, midpoint=0.9))
	plt.xlabel('gamma')
	plt.ylabel('C')
	plt.colorbar()
	plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
	plt.yticks(np.arange(len(C_range)), C_range)
	plt.title('Validation accuracy')
	plt.show()
