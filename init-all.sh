#!/bin/bash

echo -e "\033[32m 0) Fetch libsvm...\033[0m"
git submodule init
git submodule update


echo -e "\033[32m 1) Compile libsvm and add symlinks...\033[0m"
make -C libsvm clean
make -C libsvm lib

cd scratch
ln -sf ../libsvm/python/svm.py ./
ln -sf ../libsvm/python/svmutil.py ./
ln -sf ../libsvm/python/commonutil.py ./

cd ../
ln -sf libsvm/libsvm.so.2 ./


echo -e "\033[32m 2) Download datasets...\033[0m"
mkdir -p ./data/pendigits
[ -f ./data/pendigits/pendigits ] || wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits -O ./data/pendigits/pendigits
[ -f ./data/pendigits/pendigits.t ] || wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits.t -O ./data/pendigits/pendigits.t
[ -f ./data/mushrooms ] || wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms -O ./data/mushrooms
python mnist/utils.py
