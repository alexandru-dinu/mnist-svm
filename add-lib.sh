rm -f scratch/svm.py scratch/svmutil.py
rm -f ./libsvm.so.2

make -C libsvm clean
make -C libsvm lib

cd scratch
ln -s ../libsvm/python/svm.py ./
ln -s ../libsvm/python/svmutil.py ./
ln -s ../libsvm/python/commonutil.py ./

cd ../
ln -s libsvm/libsvm.so.2 ./

