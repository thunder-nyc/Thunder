# Thunder

Thunder is a [C++11](http://en.wikipedia.org/wiki/C%2B%2B11) library to provide device-transparent Tensor mathematical operations. We are currently working on a first proof-of-idea version of the library, focusing on applications in [Deep Learning](http://en.wikipedia.org/wiki/Deep_learning) using high-level parallelization and numerical optimization with GPUs and computer clusters.

Thunder is largely inspired by [Torch 7](http://torch.ch), Facebook's [TH++](http://github.com/facebook/thpp), and [EBLearn](http://eblearn.sourceforge.net)'s [libidx](http://eblearn.sourceforge.net/libidx.html). In fact, the name "Thunder" came from "TH--" if "under" could be interpreted as "--".

The library just had its development began. Contribution is very welcomed! Please contact Xiang Zhang (xiang.zhang [at] nyu.edu) for details.

## Installation

For now Thunder contains only a set of headers. The only installation instruction is to copy the directory "thunder" into your include directories, for example
```sh
cp -r thunder /usr/local/include/
```

Please note that the Thunder library is still quite new and immature. Many of its features are still under implementation. Therefore we do not yet provide a configuration or installation script. We will change this later when the library is ready to be released.

### Compilers for C++11

Thunder requires a C++11 compiler. We target our tests to [gcc](https://gcc.gnu.org) >= 4.8 and [llvm/clang](http://clang.llvm.org) >= 3.4.

## Features

Thunder has many exciting features. The following is a preview list. Some of them are already in the current public source code.

### Device Transparency

Device transparency means that we can transfer data between tensors living on different hardware seamlessly. For example
```cpp
using namespace thunder;

// Create a tensor of size 3x9x7x10
DoubleTensor cpu_tensor(3, 9, 7, 10);

// Create a tensor living on NVIDIA GPU
FloatCudaTensor gpu_tensor;

// Resize and copy from the CPU tensor
gpu_tensor.resizeAs(cpu_tensor).copy(cpu_tensor);
```

### Reference Semantics

Tensors in Thunder do not manage memory; rather, they contain [thread-safe C++11 shared pointers](http://en.cppreference.com/w/cpp/memory/shared_ptr) to underlying `Storage` objects. Unless explicitly created by constructors, static tensor creators or a call to `Tensor::clone()` for deep copying, Thunder tensors are light-weight objects that can be copied, moved or returned without heavy side effects.

That being said, we still have static memory deallocation when a `Storage` is not linked by anybody. This provides us with both fast Tensor operations and tight memory control without requiring any explicit memory calls by the user.

```cpp
using namespace thunder;

// Using tensor constructors create new underlying Storage object.
DoubleTensor tensor(3, 9, 7, 10);

// Static tensor creators also create new underlying Storage objects.
DoubleTensor created_tensor = DoubleTensor::ones(tensor.size());

// Copy constructor still points to the same Storage object.
DoubleTensor copied_tensor = tensor;

// Subtensor operators still points to the same Storage object,
// but now we have a different subtensor view of size 2x8x7x10
DoubleTensor sub_tensor = tensor[{{1,2},{1,8}}]

// However, the call to 'clone()' creates new underlying Storage.
// It is essentially a 'deep copy'.
DoubleTensor cloned_tensor = tensor.clone();
```

### Range-based `for` Loop

We support the new [C++11 range-based `for` loop](http://en.cppreference.com/w/cpp/language/range-for) on tensors. In Thunder, a range-based `for` loop iterates through the first dimension of the tensor.
```cpp
using namespace thunder;

// Create a tensor of size 3x9x7x10
DoubleTensor tensor(3, 9, 7, 10);

// Create a vector of size 10
DoubleTensor vector(10);

// Create a vector of size 7 storing result data
DoubleTensor result = DoubleTensor::zeros(7);

// Create a default blas device
Blas blas_device;

// Create a default random device
Random rand_device;

// Each t is of size 9x7x10
for (const DoubleTensor &t : tensor) {
    // Each s is of size 7x10
    for (const DoubleTensor &s : t) {
        // Do matrix-vector multiplication with vector sampled
        // from normal distribution with mean = 0 and std = 1
    	result += blas_device.gemv(s, rand_device.normal(&vector, 0, 1));
    }
}
```

### Lambda Expression

In Thunder, each tensor can accept a [lambda expression](http://en.cppreference.com/w/cpp/language/lambda) to read or change its values. The following is an example
```cpp
using namespace thunder;

// Create a tensor of size 3x9x7x10
DoubleTensor tensor(3, 9, 7, 10);

// Create a value to store the sum
double sum = 0;

// Apply a lambda that sums up the values and assign current sum to current value
tensor.apply(
   [&sum](double v) {
      sum = sum + v;
      return sum;
    });
```

### Complex Numbers

Thunder library support complex numbers natively.
```cpp
using namespace thunder;

// Create 2 tensors of size 3x9x7x10
DoubleTensor tensor1(3, 9, 7, 10);
DoubleTensor tensor2(3, 9, 7, 10);

// Construct a complex tensor using polar
DoubleComplexTensor complex_tensor = DoubleComplexTensor::polar(tensor1, tensor2);

// Get the complex norm of the tensor
DoubleTensor norm_tensor = complex_tensor.getCnrm();
```

### Serialization

We use the [boost serialization](http://www.boost.org/doc/libs/release/libs/serialization) library to serialize all data structures in Thunder.
```cpp
using namespace thunder;
using namespace boost;

// Create a tensor of size 3x9x7x10
DoubleTensor tensor(3, 9, 7, 10);

// Create a string stream
std::stringstream stream;

// Create an output archive link to the string stream
serialization::text_archive archive(stream);

// Serialize the tensor to the archive
archive << tensor;

// Now you can see the content of the serialized data
printf("Serialized data: %s\n", stream.str().c_str());
```

### Random Generators

We support all [random generators provided by the C++11 standard library](http://en.cppreference.com/w/cpp/numeric/random). They include
* Uniform distribution
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

// Create a tensor of size 3x9x7x10
DoubleTensor tensor(3, 9, 7, 10);

// Generate from gamma distribution with alpha = 1.0, beta = 1.0.
generator.gamma(1.0, 1.0, &tensor);
```

### Batch BLAS

Our BLAS routines support batch mode. The batch mode offers possiblity of speeding up BLAS routines in CPU or GPU without changing the underlying single-core implementation. This design should be more practical and easier to speed up.
```cpp
using namespace thunder;

// Create a BLAS computing device
DoubleBlas blas_device;

// Create a tensor of size 3x9x7x10
DoubleTensor tensor1(3, 9, 7, 10);

// Create another tensor of size 3x9x10
DoubleTensor tensor2(3, 9, 10);

// Computing matrix-vector multiplication in batch mode
// Now, 'result' is a tensor of size 3x9x7.
DoubleTensor result = blas_device.gemv(tensor1, tensor2);
```