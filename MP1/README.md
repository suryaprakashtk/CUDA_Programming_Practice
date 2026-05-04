```
rm -rf build
cmake -S . -B build
cmake --build build

./build/cuda_app ./data/0

ncu --set full ./build/cuda_app
```
