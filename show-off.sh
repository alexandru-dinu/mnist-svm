#!/bin/bash

# mnist
cd mnist

echo -e "\033[32m Running MNIST example...\033[0m"
python classify.py

echo -e "\n\nContinue..." && read -n 1

cd ../scratch
# mushrooms
echo -e "\033[32m Running binary classification example (mushrooms)...\033[0m"
python binary.py rbf

echo -e "\n\nContinue..." && read -n 1

# pendigits
echo -e "\033[32m Running multi-class classification example (pendigits)...\033[0m"
python multiclass.py wta rbf 0.0001

cd ../
