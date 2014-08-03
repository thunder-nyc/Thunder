Thunder
=======

Since Facebook released [th++](http://example.com/), I wanted to program a similar library addressing some of its problems by more proper template design with C++11. Originally I thought I would name the library as 'th--' just for a humour. But then I wrote '--' as 'under' and it so happens that 'thunder' is a much cooler name! What a coincidence that I used to write a 'Thunder Neural Networks' library!

The goal here to make a Torch-like tensor library that can easily manage device-dependent code using template specializations. That way, the user of this library could transfer data between devices easily, and run code on different hardwares transparently. The first stage will be for CPU and CUDA as a working proof of idea.

I chose to use C++ 11 because many new features it provides -- including but not limited to multi-threading, ranged for iterators, raw string literals and move constructors -- greatly simplifies the way we use a compiled programming language. I would say it is a very much different language now compared to what we have been taught before.

The belief is that a well-designed tensor library for C++ could be used as easily as those in scripting languages, except for some rare cases where extremely fast prototyping is needed. The immediate advantage is also tremendous -- all the C++ libraries are available to us right after a data type conversion is provided. This is quite different from a scripting language based library -- to provide bridge to extra libraries, usually a complete interface must be designed and implemented. The possibility from this alone is quite exciting -- think about what you can do with OpenCV, Qt, Redis, Boost, std::thread, and many others!

Design
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

typedef thunder::Tensor<thunder::FloatStorage> thunder::FloatTensor;
typedef thunder::Tensor<thunder::CudaFloatStorage> thunder::CudaFloatTensor;
```

As for device-dependent code, they can be easily implemented with tensor specializations as above.

Type Conversion
------
