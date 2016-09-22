
To download the released source code, please refer to our [release page on github](https://github.com/thunder-nyc/Thunder/releases).

# Thunder v0.3.0
This version adds Thunder-Linealg package, which includes linear algebra interfaces similar to that of BLAS.

## Thunder-Linalg
Thunder-Linalg  is the linalg-algebra library for thunder Tensors. Currently it consists of some linear-algebra constructors and all BLAS level 1, 2, 3 functions.

* Constructors: `diag`, `eye`, `linspace`, `logspace`, `tril`, `triu`.
* BLAS level 1: `asum`, `axpy`, `copy`, `dot`, `dotc`, `nrm2`, `rot`, `rotm`, `scal`, `swap`, `iamax`
* BLAS level 2: `gbmv`, `gemv`, `ger`, `gerc`, `hbmv`, `hemv`, `her`, `her2`, `hpmv`, `hpr`, `hpr2`, `sbmv`, `spmv`, `spr`, `spr2`, `symv`, `syr`, `syr2`, `tbmv`, `tbsv`, `tpmv`, `tpsv`, `trmv`, `trsv`.
* BLAS level 3: `gemm`, `hemm`, `herk`, `her2k`, `symm`, `syrk`, `syr2k`, `trmm`, `trsm`.

All of these operations support both real and complex tensor types, and all of them support batch mode. For example
```cpp
using namespace thunder;

// Create a BLAS computing device
// Linealg is short for linear algebra
DoubleLinalg linalg_device;

// Create a tensor of size 3x9x7x10
DoubleTensor tensor1(3, 9, 7, 10);

// Create another tensor of size 3x9x10
DoubleTensor tensor2(3, 9, 10);

// Computing matrix-vector multiplication in batch mode
// Now, 'result' is a tensor of size 3x9x7.
DoubleTensor result = linalg_device.gemv(tensor1, tensor2);
```

## Other Improvements

* Implemented a `randperm` function for Thunder-Random
* Implemented a `sort` function for Thunder-Tensor
* Implemented `viewReal` and `viewImag` functions for Thunder-Tensor
* Improve allocator usage across Thunder

## Bug fixes

* Fixed compilation error due to namespace conflict in Thunder-Exception

# Thunder v0.2.0
This version adds Thunder-Random package, which includes a rich set of random number generators.

## Thunder-Random

Thunder-Random is the random number generation engine for tensors in Thunder. We support all [random generators provided by the C++11 standard library](http://en.cppreference.com/w/cpp/numeric/random). They include

* Discrete uniform distribution
* Continuous uniform distribution
* Bernoulli distribution
* Binomial distribution
* Negative binomial distribution
* Geometric distribution
* Poisson distribution
* Exponential distribution
* Gamma distribution
* Weibull distribution
* Extreme value distribution
* Normal distribution
* Log normal distribution
* Chi squared distribution
* Cauchy distribution
* Fisher F distribution
* Student T distribution

For example
```cpp
using namespace thunder;

// Create a random number generator
DoubleRandom generator;

// Generate a tensor of size 3x9x7x10 from a gamma distribution
// with alpha = 1.0 and beta = 1.0.
DoubleTensor tensor = generator.gamma({3, 9, 7, 10}, 1.0, 1.0);
```

## Bug fixes

* Fixed extern templates for explicitly declared static functions
* Fixed compilation error using LLVM/Clang

# Thunder v0.1.0
This version adds a new serialization implementation that removes dependency on Boost.

## Thunder Serializer

Thunder now provides its own serialization functionalities that are very extensible. It can

* Serialize all fundamental types
* Avoid duplicated data saving for pointers
* Track polymorphic types and do correct serialization
* Easily extensible and non-intrusive for classes

As an example:
```cpp
using namespace thunder;

// Create a tensor of size 3x9x7x10
DoubleTensor tensor(3, 9, 7, 10);

// Create a text serializer that serializes to a string
StringTextSerializer string_serializer;

// Serialize the tensor
string_serializer.save(tensor);

// Now you can see the content of the serialized data
printf("Serialized data: %s\n", string_serializer.protocol().stream().str().c_str());
```

## CMake

From now on building Thunder will be based on CMake. The new build instructions are much cleaner with better platform independence

```sh
$ mkdir build && cd build
$ cmake .. -DBUILD_THUNDER_TESTS=ON
$ make
$ make test
```

## Other Improvements

* Added an `isUnique` function to Tensor types
* Added `extern` template declaration in top-level headers
* Bug fixes in various places

# Thunder v0.0.0

The first version of the Thunder tensor library is released. From now on the Thunder library will be put into a version controlled released cycle. The thunder version number follows the [semantic versioning](http://semver.org/) principle, in which a version is constructed in the format major.minor.patch. This major version 0 is for initial development, in which the API might be unstable.

For a preview list of features, check out our website [http://thunder.nyc](http://thunder.nyc)

## Tensor Types

Version 0.0.0 provides 5 tensor types:

* DoubleTensor
* FloatTensor
* DoubleComplexTensor
* FloatComplexTensor
* SizeTensor

Among them, the first 2 are for real valued data, and then 2 for complex data. The last one SizeTensor is for returning indices to values of tensors. It is useful for reduction operations such as max and min.

## Access and Modifying Operations

Thunder provides a rich set of access and modifying operations on tensors. They can be classified as

* Property queries: size, length, stride, dimension, storage, offset, conguity, uniqueness, etc.
* Access operators: () for reference access, [] for subtensor access.
* Iterators: subtensor iterators and reference iterators.
* Modifiers: resize, contiguous, unique
* Transformations: narrow, select, extract, shuffle, view, transpose, unfold, clone, cat, reshape
* Type conversion
* Lambda appliers

All of these access and modifying operations form the basis of tensor management.

## Element-wise Mathematical operations

Each tensor type support  a huge number of element-wise mathematical operations. They can be classified as

* Unary operations: abs, exp, log, sqrt, sin, asin, sinh, asinh, erf, tgamma, conj, fpclassify etc.
* Binary operations: add, sub, mul, div, fmod, fmin, fmax, pow, hypot, atan2, copysign etc.
* Ternary operations: polar, fma.
* Reduction operations: max, min, sum, prod, mean, var, std.
* Static constructors: zeros, ones, polars.
* Mathematical operators: +, -, *, / with both tensors and values.
