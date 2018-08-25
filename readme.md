# Cosine Similarity

The cosine similarity is a measure of similary between two vectors. Typically, it can be used as a text matching algorithm. The vector is filled by the term frequency vectors of word or sequence of X characters in text documents. To simplify the experiment, the dataset is filled with random values. In the experiment, it compute the distance between each vectors.

The same algorithm is written using different methods.

Methods:

- CPU (pure c#)
- GPU (NVidia graphic card and CUDA)
- Vectorized on CPU (using Advanced Vector Extensions. e.g. SSE, AVX, etc.)

The experiment can help better understand the advantage of using one method over another. It also provide an example of code which we can refer to when we need to.

Varying parameters:

- Method (CPU, Vectorized on CPU, GPU)
- Dataset size (number of element and number of dimension)
- Number of threads
- Array types (int/double)

# Methodology

The project build in release x64 and the option optimize code is checked.

First, it makes sure to do the JIT compilation on GPU on startup. (about 1 sec.) It run comparison for multiple matrix sizes.

It validates the result to the CPU version (1 thread) to make sure the result is the same.

# Results

Computer specs:

- SSD
- 16 GB DDR4
- NVIDIA GeForce GTX 1060 6GB (1280 cuda cores)
- Ryzen 7 (8 cores, 16 threads)

```
Integer versions:

Gpu (JIT compilation): 979 ms

Dataset: 200x100000
Simple 1 thread:        3386 ms
Simple 2 threads:       1735 ms
Simple 4 threads:       864 ms
Simple 8 threads:       476 ms
VectorizedV1 1 thread:  1420 ms
VectorizedV1 2 threads: 715 ms
VectorizedV1 4 threads: 376 ms
VectorizedV1 8 threads: 244 ms
VectorizedV2 1 thread:  982 ms
VectorizedV2 2 threads: 508 ms
VectorizedV2 4 threads: 283 ms
VectorizedV2 8 threads: 242 ms
Gpu:                    725 ms

Dataset: 2000x5000
Simple 1 thread:        17625 ms
Simple 2 threads:       8784 ms
Simple 4 threads:       4355 ms
Simple 8 threads:       2467 ms
VectorizedV1 1 thread:  7578 ms
VectorizedV1 2 threads: 3894 ms
VectorizedV1 4 threads: 2084 ms
VectorizedV1 8 threads: 1257 ms
VectorizedV2 1 thread:  6148 ms
VectorizedV2 2 threads: 3326 ms
VectorizedV2 4 threads: 1731 ms
VectorizedV2 8 threads: 1322 ms
Gpu:                    1361 ms

Dataset: 5000x25
Simple 1 thread:        930 ms
Simple 2 threads:       589 ms
Simple 4 threads:       446 ms
Simple 8 threads:       276 ms
VectorizedV1 1 thread:  626 ms
VectorizedV1 2 threads: 488 ms
VectorizedV1 4 threads: 389 ms
VectorizedV1 8 threads: 203 ms
VectorizedV2 1 thread:  5962 ms
VectorizedV2 2 threads: 3490 ms
VectorizedV2 4 threads: 1961 ms
VectorizedV2 8 threads: 1250 ms
Gpu:                    725 ms

Dataset: 1x1
Simple 1 thread:        0 ms
Simple 2 threads:       0 ms
Simple 4 threads:       0 ms
Simple 8 threads:       0 ms
VectorizedV1 1 thread:  0 ms
VectorizedV1 2 threads: 0 ms
VectorizedV1 4 threads: 0 ms
VectorizedV1 8 threads: 0 ms
VectorizedV2 1 thread:  0 ms
VectorizedV2 2 threads: 0 ms
VectorizedV2 4 threads: 0 ms
VectorizedV2 8 threads: 0 ms
Gpu:                    303 ms

Double versions:

Gpu (JIT compilation): 318 ms

Dataset: 2000x5000
Simple 1 thread:        9804 ms
Simple 2 threads:       5532 ms
Simple 4 threads:       3121 ms
Simple 8 threads:       2641 ms
VectorizedV1 1 thread:  11518 ms
VectorizedV1 2 threads: 6006 ms
VectorizedV1 4 threads: 3201 ms
VectorizedV1 8 threads: 2532 ms
VectorizedV2 1 thread:  7338 ms
VectorizedV2 2 threads: 4262 ms
VectorizedV2 4 threads: 2867 ms
VectorizedV2 8 threads: 2545 ms
Gpu:                    1614 ms

Dataset: 5000x25
Simple 1 thread:        732 ms
Simple 2 threads:       529 ms
Simple 4 threads:       358 ms
Simple 8 threads:       312 ms
VectorizedV1 1 thread:  697 ms
VectorizedV1 2 threads: 652 ms
VectorizedV1 4 threads: 321 ms
VectorizedV1 8 threads: 347 ms
VectorizedV2 1 thread:  5546 ms
VectorizedV2 2 threads: 3396 ms
VectorizedV2 4 threads: 1807 ms
VectorizedV2 8 threads: 1302 ms
Gpu:                    739 ms

Dataset: 10000x1000
Gpu:                    Exception: unspecified launch failure
```

If we consider the dataset 200x100000:

```
GpuCosineSimilarityIntegerVersionCacheKernel:
    (init):             256 ms
    (ComputeDistances): 378 ms
    (ComputeDistances): 356 ms
    (ComputeDistances): 344 ms
    (ComputeDistances): 345 ms
    (ComputeDistances): 342 ms
    (ComputeDistances): 341 ms
    (ComputeDistances): 344 ms
    (ComputeDistances): 345 ms
    (ComputeDistances): 344 ms
    (ComputeDistances): 342 ms
    (ComputeDistances): 345 ms
    (ComputeDistances): 340 ms
    (ComputeDistances): 342 ms
    (ComputeDistances): 346 ms
    (ComputeDistances): 342 ms
    (ComputeDistances): 342 ms
    (ComputeDistances): 345 ms
    (ComputeDistances): 343 ms
    (ComputeDistances): 366 ms
    (ComputeDistances): 345 ms
    (ComputeDistances): 356 ms
    (ComputeDistances): 420 ms
    (ComputeDistances): 346 ms
    (ComputeDistances): 343 ms
    (ComputeDistances): 342 ms
    (ComputeDistances): 344 ms
    (ComputeDistances): 344 ms
    (ComputeDistances): 349 ms
    (ComputeDistances): 346 ms
    (ComputeDistances): 343 ms
    (ComputeDistances): 354 ms
    (ComputeDistances): 344 ms
    (ComputeDistances): 342 ms
    (ComputeDistances): 343 ms
    (ComputeDistances): 348 ms
    (ComputeDistances): 343 ms
    (ComputeDistances): 355 ms
    (ComputeDistances): 343 ms
    (ComputeDistances): 351 ms
    (ComputeDistances): 352 ms
    (ComputeDistances): 342 ms
    (ComputeDistances): 346 ms
    (ComputeDistances): 346 ms
    (ComputeDistances): 344 ms
    (ComputeDistances): 346 ms
    (ComputeDistances): 344 ms
    (ComputeDistances): 343 ms
    (ComputeDistances): 343 ms
    (ComputeDistances): 345 ms
    (ComputeDistances): 344 ms
    (ComputeDistances): 346 ms
    (ComputeDistances): 346 ms
    (ComputeDistances): 344 ms
    (ComputeDistances): 343 ms
    (ComputeDistances): 351 ms
    (ComputeDistances): 347 ms
    (ComputeDistances): 344 ms
    (ComputeDistances): 354 ms
    (ComputeDistances): 345 ms
    (ComputeDistances): 343 ms
    min: 340 ms
    max: 420 ms
    (dispose):          34 ms
```

If we consider the dataset: 200x100000:

```
Simple 1 thread:        3338 ms
Simple 2 threads:       1694 ms
Simple 4 threads:       845 ms
Simple 8 threads:       446 ms
SimpleV2 1 thread:      1940 ms
SimpleV2 2 threads:     982 ms
SimpleV2 4 threads:     549 ms
SimpleV2 8 threads:     273 ms
```

# Conclusion

With the simple method, adding more thread reduce the duration.

There is a minimal cost to communicate with the GPU device (about 300 ms in the experimentation and only occur on the first GPU call). You need to have a great amount of data to use the GPU. Otherwise, it's slower than the single thread version. The communication cost with GPU is negligible when using large arrays. If the array is too large, we got an exception. (Maybe it's time to do batch processing and do multiple GPU call.)

The Advanced Vector Extensions of modern CPU can be used per thread. Adding more threads reduce the computation time. Compared to the simple method, it uses about half (or less) the time to do the same job in the integer version. If the dataset is a double array, the performance is the same or worst.

Obviously, using double is way slower than integer. If possible, always prefer integer. If you want to keep some digits, you could multiple the number by 10 or 100 and convert it to integer. If you really want to keep double, maybe you should consider using the GPU.

If we compare the vectorized version (integer array, v1 and v2), the dot product is faster than doing an addition/multiplication on an accumulator vector and taking the sum of the accumulator when having small dimension in the array. (It's slower than the simple method on 1 thread.) But, if you consider an array with a lot of dimension, it's faster using an accumulator vector than using the dot product operation.

The kernel function can be cached for multiple use. If we consider the dataset 200x100000:
- Initialization can take 256 ms.
- Computing the distance vary between 340 ms and 420 ms. (about a variation of 80 ms)

Precalculating the magnitude for each vector greatly reduce the amount of operations to do. (class: SimpleV2CosineSimilarityIntegerVersion)

# Copyright and license

Code released under the MIT license.
