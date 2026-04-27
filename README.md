# CUDA_Programming_Practice
Quick command references

```
nvcc -o out <.cu file> -l<library name>
nvcc -o out add.cu -lcublas
./out
```


# Profiling Commands
## Using the profiling tools

```
// System level statistics like time for API calls
nsys profile --trace=cuda --stats=true -o report --force-overwrite=true ./out

// Kernal level memory usage
ncu --set full ./out
```

`starter_code.cu` has template to use for practice assignments

