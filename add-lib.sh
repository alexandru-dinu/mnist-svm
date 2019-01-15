make -C libsvm clean
make -C libsvm lib

cd scratch
ln -sf ../libsvm/python/svm.py ./
ln -sf ../libsvm/python/svmutil.py ./
ln -sf ../libsvm/python/commonutil.py ./

cd ../
ln -sf libsvm/libsvm.so.2 ./

