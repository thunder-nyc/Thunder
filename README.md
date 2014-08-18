# Thunder

Thunder is a [C++11](http://en.wikipedia.org/wiki/C%2B%2B11) library to provide device-transparent Tensor mathematical operations. We are currently working on a first proof-of-idea version of the library, focusing on applications in [Deep Learning](http://en.wikipedia.org/wiki/Deep_learning) using high-level parallelization and numerical optimization with GPUs and computer clusters.

Thunder is largely inspired by [Torch 7](http://torch.ch) and Facebook's [TH++](http://github.com/facebook/thpp). In fact, the name "Thunder" came from "TH--" if "under" could be interpreted as "--".

The library just had its development began. Contribution is very welcomed! Please contact Xiang Zhang (xiang.zhang [at] nyu.edu) for details.

## Installation

For now Thunder contains only a set of headers. The only installation instruction is to copy the directory "thunder" into your include directories, for example
```sh
cp -r thunder /usr/local/include/
```

Please note that the Thunder library is still quite new and immature. Many of its features are still under implementation. Therefore we do not yet provide a configuration or installation script. We will change this later when the library is ready to be released.

## Features

Thunder has many exciting features. The following is a preview list. Some of them are already in the current source code tree.

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

### Ranged `for` Loop

We support the new [C++11 ranged `for` loop](http://en.cppreference.com/w/cpp/language/range-for) on tensors. In Thunder, a ranged `for` loop iterates through the first dimension of the tensor.
```cpp
using namespace thunder;

// Create a tensor of size 3x9x7x10
DoubleTensor tensor(3, 9, 7, 10);

// Each t is of size 9x7x10
for (DoubleTensor t : tensor) {
    // Each s is of size 7x10
    for (DoubleTensor s : t) {
    	// Do something with s and t	
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
* normal distribution
* Log normal distribution
* Chi squared distribution
* Cauchy distribution
* Fisher F distribution
* Student T distribution

For example
```cpp
using namespace thunder;

// Create a tensor of size 3x9x7x10
DoubleTensor tensor(3, 9, 7, 10);

// Generate from gamma distribution with alpha = 1.0, beta = 1.0.
tensor.gamma(1.0, 1.0);
```

### Serialization

We use the boost serialization library to serialize all data structures in Thunder.
```cpp
using namespace thunder;
using namespace boost;

// Create a tensor of size 3x9x7x10
DoubleTensor tensor(3, 9, 7, 10);

// Create an output archive link to file tensor.tdt
serialization::text_archive archive(std::ofstream("tensor.tdt"));

// Serialize to the file
archive << tensor;
```
