```
rm -rf build
cmake -S . -B build
cmake --build build

./build/cuda_app

ncu --set full ./build/cuda_app
```
