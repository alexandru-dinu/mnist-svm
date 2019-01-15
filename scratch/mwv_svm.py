import numpy as np
from tqdm import tqdm

import svmutil

num_classes = 10
num_features = 16


# MWV_SVM = max wins voting (one-vs-one)

def mwv_prepare_datasets(ys, xs):
    ys_by_classes = {cls: [] for cls in range(num_classes)}
    xs_by_classes = {cls: [] for cls in range(num_classes)}

    for cls in range(num_classes):
        for y, x in zip(ys, xs):
            if y == cls:
                ys_by_classes[cls] += [y]
                xs_by_classes[cls] += [x]

    return ys_by_classes, xs_by_classes


def mwv_construct_classifiers(ys_by_classes, xs_by_classes, ktype, gamma):
    classifiers = []

    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            ys_ij = ys_by_classes[i] + ys_by_classes[j]
            xs_ij = xs_by_classes[i] + xs_by_classes[j]

            print("\tConstructing SVM for pair {}x{}".format(i, j))

            svm_params = "-t " + str(ktype) + " -g " + str(gamma) + " -q"
            m_ij = svmutil.svm_train(ys_ij, xs_ij, svm_params)
            classifiers.append(m_ij)

    return classifiers


def mwv_predict_single(classifiers, y, x):
    votes = [0] * num_classes

    for c in classifiers:
        p_label, p_acc, p_val = svmutil.svm_predict([y], [x], c, options='-q')

        pred_class = int(p_label[0])

        votes[pred_class] += 1

    return np.argmax(votes)


def mwv_accuracy(classifiers, ys, xs):
    correct = 0
    num_test = len(ys)

    for i in tqdm(range(num_test)):
        y, x = ys[i], xs[i]

        pred = mwv_predict_single(classifiers, y, x)

        if pred == y:
            correct += 1

    return round(100.0 * correct / num_test, 5)


def mwv_svm(ys_test, xs_test, ys_train, xs_train, ktype, gamma):
    i2k = {
        2: 'g',
        1: 'p'
    }

    print("MWV_SVM, ktype = {}, gamma = {}".format(i2k[ktype], gamma))

    print("Preparing datasets...")
    ys_by_classes, xs_by_classes = mwv_prepare_datasets(ys_train, xs_train)

    print("Constructing classifiers...")
    classifiers = mwv_construct_classifiers(ys_by_classes, xs_by_classes, ktype, gamma)

    print("Evaluating model...")
    acc = mwv_accuracy(classifiers, ys_test, xs_test)

    print("MWV_SVM ACC = {}".format(acc))
