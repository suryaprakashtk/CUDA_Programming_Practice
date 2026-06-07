#!/bin/bash
rm -rf build
cmake -S . -B build
echo "--------------";
cmake --build build
echo "--------------";

for i in 0 1 2 3 4 5 6 7 8 9;
do
	echo "--------------";
	echo "Dataset " $i 
	./build/cuda_app ./data/${i}
done