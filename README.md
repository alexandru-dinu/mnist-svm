# Classification tests using SVM

To create `libsvm` bindings, first run `add-lib.sh`.
The script will compile the shared library `libsvm.so` and will generate the necessary python symlinks.

## Datasets used

- **pendigits**
    - source: [UCI](http://www.ics.uci.edu/~mlearn/MLRepository.html) / Pen-Based Recognition of Handwritten Digits Data Set
    - number of classes: 10
    - number of examples: 7,494 / 3,498 (testing)
    - number of features: 16
    - [pendigits](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits)
    - [pendigits.t (testing)](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits.t)
    
- **mushrooms**
    - source: [UCI](http://www.ics.uci.edu/~mlearn/MLRepository.html) / mushrooms
    - Preprocessing: Each nominal attribute is expaned into several binary attributes. The original attribute #12 has missing values and is not used.
    - number of classes: 2
    - number of examples: 8124
    - number of features: 112
    - [mushrooms](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms)
    
- **MNIST**
    - [train-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz):  training set images (60000 examples)
    - [train-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz):  training set labels 
    - [t10k-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz):   test set images (10000 examples)
    - [t10k-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz):   test set labels