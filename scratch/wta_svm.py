import numpy as np
from tqdm import tqdm

import svmutil

num_classes = 10
num_features = 16


# WTA_SVM = winner takes all (one-vs-all)

def wta_construct_classifier(positive_class, ys, xs, ktype, gamma):
    # create custom labels for this classifier
    custom_ys = [y if y == positive_class else -1 for y in ys]

    svm_params = "-t " + str(ktype) + " -g " + str(gamma) + " -q"
    m = svmutil.svm_train(custom_ys, xs, svm_params)

    return m


def wta_construct_classifiers(ys, xs, ktype, gamma):
    classifiers = {}

    for cls in range(num_classes):
        print("\tConstructing SVM for class {}".format(cls))
        classifiers[cls] = wta_construct_classifier(cls, ys, xs, ktype, gamma)

    return classifiers


def wta_predict_single(classifiers, y, x):
    vals = [None] * num_classes

    for cls in range(num_classes):
        p_label, p_acc, p_val = svmutil.svm_predict(
            [y], [x], classifiers[cls], options='-q')

        decision_val = p_val[0][0]

        # [pos, neg]
        pos_cls, neg_cls = classifiers[cls].get_labels()

        vals[cls] = decision_val if pos_cls != -1 else -decision_val

    return np.argmax(vals)


def wta_accuracy(classifiers, ys, xs):
    num_test = len(ys)
    correct = 0

    for i in tqdm(range(num_test)):
        y, x = ys[i], xs[i]

        pred = wta_predict_single(classifiers, y, x)
        if pred == y:
            correct += 1

    return round(100.0 * correct / num_test, 5)


def wta_svm(ys_test, xs_test, ys_train, xs_train, ktype, gamma):
    i2k = {
        2: 'g',
        1: 'p'
    }

    print("WTA_SVM, ktype = {}, gamma = {}".format(i2k[ktype], gamma))

    # construct a binary SVM for each class (class -> SVM)
    print("Constructing classifiers...")
    classifiers = wta_construct_classifiers(ys_train, xs_train, ktype, gamma)

    print("Evaluating model...")
    acc = wta_accuracy(classifiers, ys_test, xs_test)

    print("WTA_SVM ACC = {}".format(acc))
