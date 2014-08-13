Thunder
=======

Thunder, ak.k. TH--, a joking name for Facebook's [TH++](https://github.com/facebook/thpp) and [Torch](http://torch.ch)'s libTH.

Thunder is an effort to design and implement an [Torch](http://torch.ch)-like environment for high-dimensional numerical operations on tensors in C++11. This problem can be decomposed into 3 parts: device-dependent memory management, platform-dependent numerics, and object serialization. Our current goal is to provide a working proof of idea using the Eigen library and CUDA library as device-dependent numerical backends.

This code shows what we want to do at a very high level, using deep learning as an example:
```cpp
// Thunder library (a.k.a. TH--, or "TH under")
using namespace thunder;

// Construct a model in double tensor
nn::Sequential<DoubleTensor> model;

// Compiler will automatically deduce nn::Linear<DoubleTensor>
model.Add(nn::Linear(1024, 10));
model.Add(nn::SoftMax());

// Construct a model in CUDA and initialize from a different type
nn::Sequential<CudaFloatTensor> cu_model(model);
```


Managing Device-Dependent Code
------

One problem of th++ is that the storage class is templated and templated only on basic data types, therefore it is very difficult to manage device-dependent code. The reason is that in C++, device-dependent platforms such as CUDA and OpenCL use these basic data types (e.g., float and double) with no difference from CPU code. If there is no way to distinguish between devices using these tensor types, then it is very complicated to write device-dependent numerical computations and switch between them transparently.

However, for a very long time C++ already has a solution to this problem -- allocators. Allocators are used to provide a memory-model that is device dependent for any data type of interest. The C++ Standard Template Library (STL) uses them, and a similar approach can be found in another project -- the [Thrust](http://thrust.github.io) library. Therefore, the solution to the problem of th++ is very easy: instead of templating only on the basic datatypes, also template on an allocator and provide a default one.

Using this idea, the storage class could be implemented as
```cpp
template <typename T, typename Allocator = std::allocator<T> >
class thunder::Storage{/* Implementation here */};
```

Then, we can easily give typedefs like
```cpp
typedef thunder::Storage<float> thunder::FloatStorage;
typedef thunder::Storage<float, thrust::system::cuda::allocator<float> > thunder::CudaFloatStorage;
```

As a result, we could implement the tensor library with
```cpp
template<typename Storage>
class thunder::Tensor{/* Implementation here */};
```

As for device-dependent code, they can be easily implemented with tensor specializations as above.

Managing Platform-Dependent Numerical Library
------

Previously we show a design of templating the Tensor class on the Storage class so that they can be specialized to different computing devices. However, even on a single device, we have many numerical libraries to use. Especially, in CPU for which the Storage classes might be the same, we have choices between [Eigen](http://eigen.tuxfamily.org/), [BLAS](http://www.netlib.org/blas)/[LAPACK](http://www.netlib.org/lapack)(including OpenBLAS, ATLAS, etc.), and [Intel MKL](https://software.intel.com/en-us/intel-mkl). All these stack of libraries all provide almost the same functionality. It is up to the programmer to choose which to use at compile time.

To hide this complication, we can wrap these functions in a inheritance structure. For example
```cpp
template < typename Storage >
class EigenTensor< Storage > : public Tensor < Storage >
```

Serialization
------

Thunder library will rely on [BOOST-serialization](www.boost.org/doc/libs/release/libs/serialization) to do serialization automatically.