# Thunder-Tensor

Thunder-Tensor is the most important package of the Thunder project. It consists of a tensor class that functions as a multi-dimensional array that can be used in various mathematical operations. To use the Thunder-Tensor package, you just need to include the `thunder/tensor.hpp` header. Tensors in Thunder-tensor are row-major.

```cpp
#include <thunder/tensor.hpp>
```

## Basic tensor types

The base of tensor class is declared as

```cpp
namespace thunder {
namespace tensor {
  template < typename S = DoubleStorage > class Tensor;
}  // namespace tensor
template < typename S = DoubleStorage > using Tensor = tensor::Tensor < S >;
}  // namespace thunder
```

where `S` is some storage class followng the design of the Thunder-Storage package. Although the implementation of `class Tensor` is in the `::thunder::tensor` namespace, as in the above it is also introduced to the `::thunder` namespace for your convenience.

5 tensor types are externally template-declared and pre-compiled in both `::thunder::tensor` and `::thunder` namespaces, so that you can use them without prolonged compilation time. The following table details them.

| Name                | Definition                     |
|---------------------|--------------------------------|
| SizeTensor          | Tensor< SizeStorage >          |
| DoubleTensor        | Tensor< DoubleStorage >        |
| FloatTensor         | Tensor< FloatStorage >         |
| DoubleComplexTensor | Tensor< DoubleComplexStorage > |
| FloatComplexTensor  | Tensor< FloatComplexStorage >  |

Note that `Tensor` without template parameter is equivalent to `DoubleTensor` because by default it uses `DoubleStorage`.

For example
```cpp
#include <thunder/tensor.hpp>
using namespace thunder;
// Create a double complex tensor of size 3 x 9 x 7
DoubleComplexTensor tensor(3, 9, 7);
```

## Symbols defined in Tensor class

In this section,  assume that the storage template parameter is `S`.

The following symbols are defined in each of the tensor class from the Thunder-Storage package.

| Name            | Type definition             |
|-----------------|-----------------------------|
| storage_type    | S                           |
| allocator_type  | typename S::allocator_type  |
| value_type      | typename S::value_type      |
| reference       | typename S::reference       |
| const_reference | typename S::const_reference |
| difference_type | typename S::difference_type |
| size_type       | typename S::size_type       |
| pointer         | typename S::pointer         |
| const_pointer   | typename S::const_pointer   |

The following symbols are defined in each of the tensor class for tensor data members.

| Name            | Type definition                        |
|-----------------|----------------------------------------|
| size_storage    | Storage< size_type >                   |
| stride_storage  | Storage< difference_type >             |
| storage_pointer | ::std::shared_ptr< S > storage_pointer |
| dim_type        | typename size_storage::size_type       |

The following symbols are defined in each of the tensor class for hanling complex numbers.

| Name         | Type definition                                      |
|--------------|------------------------------------------------------|
| real_storage | typename StorageType< S >::real_storage real_storage |
| real_tensor  | Tensor< real_storage >                               |
| real_type    | typename real_storage::value_typde                   |

Note that `template < typename S > class StorageType;` is defined in `thunder/tensor/storage_type.hpp`. It has a member `real_storage` that is defined as `S` if `S` is some real-value storage type, or as the corresponding real-value storage if `S` is some complex-value storage type.

## Constructors and Destructors

### explicit Tensor(allocator_type alloc = allocator_type());

Default constructor with allocator. It creates a tensor of size 1. Thunder-Tensor package does not support zero-size tensors.

#### Exceptions

None.

### Tensor(size_storage sz, allocator_type alloc = allocator_type());

Constructor with size storage. For all dimensions, the size parameter must be larger than 0.

Because this constructor is non-explicit, it is possible to pass size storage to construct a tensor implicitly. Also, because the storage constructor that accepts initializer list is also non-explicit, a tensor can also be constructed implicitly from an initializer list. The storage or initializer list in this case is used as size parameters.

#### Exceptions

`invalid_argument` if the size storage is empty or the size of any dimension is 0.

### explicit Tensor(size_type sz0, allocator_type alloc = allocator_type());
### explicit Tensor(size_type sz0, size_type sz1, allocator_type alloc = allocator_type());
### explicit Tensor(size_type sz0, size_type sz1, size_type sz2, allocator_type alloc = allocator_type());
### explicit Tensor(size_type sz0, size_type sz1, size_type sz2, size_type sz3, allocator_type alloc = allocator_type());

Convenient constructors for up to 4 dimensions.

#### Exceptions

`invalid_argument` if any of the size parameters is zero.

### explicit Tensor(storage_pointer s, size_type os = 0);

Construct tensor using the storage `s` and offset `os`. It will create a 1-dimensional tensor with size equal to `s->size() - os`, which should be larger than 0.

#### Exceptions

`invalid_argument` if the storage pointer is `nullptr`.

`out_of_range` if the offset exceed the size of the storage.

### Tensor(size_storage sz, storage_pointer s, size_type os = 0);

Construct tensor using the size in `sz`, the shared storage `s` and the offset `os`. `s->size() - os` must be larger than or equal to the product of all values in `sz`. For all dimensions, the size parameter must be larger than 0.

#### Exceptions

`invalid_argument` if the size storage is empty, the size of any dimension is 0 or the storage pointer is `nullptr`.

`out_of_range` if the offset or the size exceed the storage size.

### Tensor(size_storage sz, stride_storage st, allocator_type alloc = allocator_type());

Construct tensor using the size in `sz` and stride in `st`. For all dimensions, the size parameter must be larger than 0. For all dimensions, the stride parameter can be negative, positive or 0. A minimum storage whose size that can span all indexed values will be properly created, the the offset of the first value in the tensor will be properly set.

The stride is a storage whose value at each dimension is multiplied with the corresponding index to compute the offset used to index a value. For example, for a 3-D tensor with stride '{st0, st1, st2}' the value at index `{i, j, k}` is computed as the value at `i * st0 + j * st1 + st2` plus some offset in the internal storage array.

#### Exceptions

`invalid_argument` if the size storage is empty, the stride storage is empty, the size of any dimension is 0 or the size storage and stride storage have different dimensions.

### Tensor(size_storage sz, stride_storage st, storage_pointer s, size_type os = 0);

Construct tensor using the size in `sz`, the stride in `st`, the shared storage `s` and offset `os`. For all dimensions, the size parameter must be larger than 0. For all dimensions, the stride parameter can be negative, positive or 0. `os` must be larger than the largest absolute value of any possible negative index value to the storage. `s->size()` must be larger than `os` plus the largest possible positive index value to the storage.

The stride is a storage whose value at each dimension is multiplied with the corresponding index to compute the offset used to index a value. For example, for a 3-D tensor with stride '{st0, st1, st2}' the value at index `{i, j, k}` is computed as the value at `i * st0 + j * st1 + st2` plus some offset in the internal storage array.

#### Exceptions

`invalid_argument` if the size storage is empty, the stride storage is empty, the size of any dimension is 0, the size storage and stride storage have different dimensions, or the size of any dimension is 0.

### Tensor(const Tensor &y);

Copy-constructor.

#### Exceptions

None.

### Tensor(Tensor &&y);

[Move-constructor](http://en.cppreference.com/w/cpp/language/move_constructor).

#### Exceptions

None.

### template < typename Other_S > explicit Tensor(const Tensor< Other_S > &y, allocator_type alloc = allocator_type());

Templated type-conversion constructor. The values are converted to the newly constructed tensor using `static_cast`. If a complex tensor is converted to a real tensor, only its real part will be used.

#### Exceptions

None.

### ~Tensor();

The desctructor.

#### Exceptions

None.

## Type conversion

```cpp
#include <thunder/tensor.hpp>
using namespace thunder;

// Create a complex tensor of size 3 x 9 x 7
DoubleComplexTensor x(3, 9, 7);
// Convert the real part of x to a double tensor
DoubleTensor y1 = x.template type< DoubleTensor >();
// The same thing using static function
DoubleTensor y2 = DoubleComplexTensor::template type< DoubleTensor >(x);
```

### template < typename T > T type(typename T::allocator_type alloc = typename T::allocator_type() const;

Convert this tensor to a new tensor of type `T`, in which each value is converted via `static_cast`. If this tensor is complex and `T` is real, only the real part of each complex value is converted.

#### Exceptions

None.

### template < typename T > static T type(const Tensor &x, typename T::allocator_type alloc = typename T::allocator_type() const;

Static version of the type conversion function. Convert the tensor `x` to type `T`.

#### Exceptions

None.

## Lambda functions

```cpp
#include <thunder/tensor.hpp>
using namespace thunder;

// Create a tensor of size 7
DoubleTensor x(7);
// Compute the cumulative sum
double sum1 = 0;
x.apply([&sum](double *v) {sum2 += *v; *v = sum1;});
// The same thing using static call; but the values in x is not changed.
double sum2 = 0;
y = DoubleTensor::apply(x, [&sum](double *v) {sum2 += *v; *v = sum2;});
```

### const Tensor& apply(const ::std::function< value_type(value_type) > &lambda) const;
### Tensor& apply(const ::std::function< value_type(value_type) > &lambda);

Apply the function `lambda` to each value in this tensor and change the old value to the returned value.

#### Exceptions

None except in `lambda`.

### static Tensor apply(const Tensor &x, const ::std::function< value_type(value_type) > &lambda);

Apply the function `lambda` to each value of tensor `x` and store the new value to a new tensor. The new tensor is returned.

#### Exceptions

None except in `lambda`.

### const Tensor& apply(const ::std::function< value_type(const value_type&) > &lambda) const;
### Tensor& apply(const ::std::function< value_type(const value_type&) > &lambda);

Apply the function `lambda` to each value in this tensor and change the old value to the returned value.

#### Exceptions

None except in `lambda`.

### static Tensor apply(const Tensor &x, const ::std::function< value_type(const value_type&) > &lambda);

Apply the function `lambda` to each value of tensor `x` and store the new value to a new tensor. The new tensor is returned.

#### Exceptions

None except in `lambda`.

### const Tensor& apply(const ::std::function< void(value_type&) > &lambda) const;
### Tensor& apply(const ::std::function< void(value_type&) > &lambda);

Apply the function `lambda` to each value in this tensor. The function `lambda` may change the old value as a reference.

#### Exceptions

None except in `lambda`.

### static Tensor apply(const Tensor &x, const ::std::function< void(value_type&) > &lambda);

Apply the function `lambda` to each value of tensor `x` and store the changed value to a new tensor. The new tensor is returned. The function may change the old value as a reference.

#### Exceptions

None except in `lambda`.

### const Tensor& apply(const ::std::function< void(value_type*) > &lambda) const;
### Tensor& apply(const ::std::function< void(value_type*) > &lambda);

Apply the function `lambda` to each value in this tensor. The function `lambda` may change the old value as a pointer.

#### Exceptions

None except in `lambda`.

### static Tensor apply(const Tensor &x, const ::std::function< void(value_type*) > &lambda);

Apply the function `lambda` to each value of tensor `x` and store the changed value to a new tensor. The new tensor is returned. The function `lambda` may change the old value as a pointer.

#### Exceptions

None except in `lambda`.

## Property query

```cpp
#include <cstdio>
#include <thunder/tensor.hpp>
using namespace std;
using namespace thunder;

// Create a tensor of size 3 x 9 x 7
DoubleTensor x(3, 9, 7);

// Print a set of properties
printf("Dimension: %ld\n", x.dimension());
printf("Size: %ld x %ld x %ld\n", x.size(0), x.size(1), x.size(2));
printf("Length: %ld\n", x.length());
printf("Stride: %ld x %ld x %ld\n", x.stride(0), x.stride(1), x.stride(2));
printf("Offset: %ld\n", x.offset());
printf("Data: %p\n", x.data());
printf("Value at {2, 5, 3}: %f\n", x.get(2, 5, 3));
printf("Contiguity: %s\n", x.isContiguous() ? "true" : "false");
printf("Unique: %s\n", x.isUnique() ? "true" : "false");

// The same thing using static functions
printf("Dimension: %ld\n", DoubleTensor::dimension(x));
printf("Size: %ld x %ld x %ld\n", DoubleTensor::size(x, 0), DoubleTensor::size(x, 1), DoubleTensor::size(x, 2));
printf("Length: %ld\n", DoubleTensor::length(x));
printf("Stride: %ld x %ld x %ld\n", DoubleTensor::stride(x, 0), DoubleTensor::stride(x, 1), DoubleTensor::stride(x, 2));
printf("Offset: %ld\n", DoubleTensor::offset(x));
printf("Data: %p\n", DoubleTensor::data(x));
printf("Value at {2, 5, 3}: %f\n", DoubleTensor::get(x, 2, 5, 3));
printf("Contiguity: %s\n", DoubleTensor::isContiguous(x) ? "true" : "false");
printf("Unique: %s\n", DoubleTensor::isUnique(x) ? "true" : "false");
```

### template < typename T > bool isSameSizeAs(const T &y) const;

Return whether this tensor as the same size as the tensor `y`.

#### Exceptions

None.

### static template < typename T > bool isSameSizeAs(const Tensor &x, const T &y);

Return whether the tensor `x` is the same size as the tensor `y`.

#### Exceptions

None.

### dim_type dimension() const;

Get the dimension of this tensor.

#### Exceptions

None.

### static dim_type dimension(const Tensor &x);

Get the dimension of tensor `x`.

#### Exceptions

None.

### size_storage size() const;

Get the size as a storage for this tensor. The returned storage has size equal to the dimension of the tensor.

#### Exceptions

None.

### static size_storage size(const Tensor &x);

Get the size as a storage for tensor `x`. The returned storage has size equal to the dimension of the tensor.

#### Exceptions

None.

### size_type size(dim_type dim) const;

Get the size at dimension `dim` for this tensor. `dim` can be any integer from 0 to `this->dimension() - 1`.

#### Exceptions

`out_of_range` if the input dimension exceeds tensor dimension.

### static size_type size(const Tensor &x, dim_type dim);

Get the size at dimension `dim` for tensor `x`. `dim` can be any integer from 0 to `x.dimension() - 1`.

#### Exceptions

`out_of_range` if the input dimension exceeds tensor dimension.

### size_type length() const;

Get the length of this tensor. The length is the product of the size in all dimensions.

#### Exceptions

None.

### static size_type length(const Tensor &x);

Get the length of tensor `x`. The length is the product of the size in all dimensions.

#### Exceptions

None.

### stride_storage stride() const;

Get the stride as a storage for this tensor. The returned storage has size equal to the dimension of the tensor.

#### Exceptions

None.

### static stride_storage stride(const Tensor &x);

Get the stride as a storage for tensor `x`. The returned storage has size equal to the dimension of the tensor.

#### Exceptions

None.

### difference_type stride(dim_type dim) const;

Get the stride at dimension `dim` for this tensor. `dim` can be any integer from 0 to `this->dimension() - 1`.

#### Exceptions

`out_of_range` if the input dimension exceeds tensor dimension.

### static difference_type stride(const Tensor &x, dim_type dim) const;

Get the stride at dimension `dim` for tensor `x`. `dim` can be any integer from 0 to `x.dimension() - 1`.

#### Exceptions

`out_of_range` if the input dimension exceeds tensor dimension.

### storage_pointer storage() const;

Get the shared pointer of the storage associated with this tensor.

#### Exceptions

None.

### static storage_pointer storage(const Tensor &x) const;

Get the shared pointer of the storage associated with tensor `x`.

#### Exceptions

None.

### size_type offset() const;

Get the offset value of this tensor. The offset indicates the location of the first tensor value in its associated storage.

#### Exceptions

None.

### static size_type offset(const Tensor &x);

Get the offset value of tensor `x`. The offset indicates the location of the first tensor value in its associated storage.

#### Exceptions

None.

### pointer data() const;

Get the pointer to the first value of this tensor. It points to the location in the storage indexed by the tensor's offset.

#### Exceptions

None.

### static pointer data(const Tensor &x);

Get the pointer to the first value of tensor `x`. It points to the location in the storage indexed by the tensor's offset.

#### Exceptions

None.

### allocator_type allocator() const;

Get the allocator of this tensor.

#### Exceptions

None.

### static allocator_type allocator(const Tensor &x);

Get the allocator of the tensor `x`.

#### Exceptions

None.

### reference get() const;

Get the reference to the first value of this tensor.

#### Exceptions

None.

### static reference get(const Tensor &x);

Get the reference to the first value of tensor `x`.

#### Exceptions

None.

### reference get(const size_storage &pos) const;

Get the reference to the value indexed by `pos` for this tensor. When the tensor has more dimensions, all indices of other dimensions are assumed to be 0.

#### Exceptions

`out_of_range` if position exceeds tensor size.

### static reference get(const Tensor &x, const size_storage &pos);

Get the reference to the value indexed by `pos` for tensor `x`. When the tensor has more dimensions, all indices of other dimensions are assumged to be 0.

#### Exceptions

`out_of_range` if position exceeds tensor size.

### reference get(size_type pos) const;

Get the reference to the value indexed by `pos` for this tensor. When the tensor has more dimensions, all indices of other dimensions are assumed to be 0.

#### Exceptions

`out_of_range` if position exceeds tensor size.

### static reference get(const Tensor &x, size_type pos);

Get the reference to the value indexed by `pos` for tensor `x`. When the tensor has more dimensions, all indices of other dimensions are assumged to be 0.

#### Exceptions

`out_of_range` if position exceeds tensor size.

### reference get(size_type pos0, size_type pos1) const;

Get the reference to the value indexed by `{pos0, pos1}` for this tensor. When the tensor has more dimensions, all indices of other dimensions are assumed to be 0.

#### Exceptions

`out_of_range` if position exceeds tensor size.

### static reference get(const Tensor &x, size_type pos0, size_type pos1);

Get the reference to the value indexed by `{pos0, pos1}` for tensor `x`. When the tensor has more dimensions, all indices of other dimensions are assumed to be 0.

#### Exceptions

`out_of_range` if position exceeds tensor size.

### reference get(size_type pos0, size_type pos1, size_type pos2) const;

Get the reference to the value indexed by `{pos0, pos1, pos2}` for this tensor. When the tensor has more dimensions, all indices of other dimensions are assumed to be 0.

#### Exceptions

`out_of_range` if position exceeds tensor size.

### static reference get(const Tensor &x, size_type pos0, size_type pos1, size_type pos2);

Get the reference to the value indexed by `{pos0, pos1, pos2}` for tensor `x`. When the tensor has more dimensions, all indices of other dimensions are assumed to be 0.

#### Exceptions

`out_of_range` if position exceeds tensor size.

### reference get(size_type pos0, size_type pos1, size_type pos2, size_type pos3) const;

Get the reference to the value indexed by `{pos0, pos1, pos2, pos3}` for this tensor. When the tensor has more dimensions, all indices of other dimensions are assumed to be 0.

#### Exceptions

`out_of_range` if position exceeds tensor size.

### static reference get(const Tensor &x, size_type pos0, size_type pos1, size_type pos2, size_type pos3);

Get the reference to the value indexed by `{pos0, pos1, pos2, pos3}` for tensor `x`. When the tensor has more dimensions, all indices of other dimensions are assumed to be 0.

#### Exceptions

`out_of_range` if position exceeds tensor size.

### bool isContiguous() const;

Return whether this tensor is contiguous. Contiguity is defined as `this->stride(i) == this->size(i - 1) * this->stride(i)`, for `i` between 1 and `this->dimension() - 1`, and `this->stride(this->dimension() - 1) == 1`.

#### Exceptions

None.

### static bool isContiguous(const Tensor &x) const;

Return whether tensor `x` is contiguous. Contiguity is defined as `x.stride(i) == x.size(i + 1) * x.stride(i + 1)`, for `i` between 0 and `x.dimension() - 2`, and `x.stride(x.dimension() - 1) == 1`.

#### Exceptions

None.

### bool partialContiguity(dim_type a, dim_type b) const;

Return whether this tensor is partial contiguity between dimension `a` and dimension `b`. Partial contiguity is defined as `this->stride(i) == this->size(i + 1) * this->stride(i + 1)` for `i` between `a` and `::std::min(b - 1, this->dimension() - 1)`. If `b >= this->dimension()`, there is the additional requirement that `this->stride(this->dimension() - 1) == 1`.

As a result, `this->partialContiguity(0, this->dimension())` is equivalent to `this->isContiguous()`. However, `this->partialContiguity(0, this->dimension() - 1)` is different because it does not require that `this->stride(this->dimension() - 1) == 1`.

#### Exceptions

None.

### static bool partialContiguity(const Tensor &x, dim_type a, dim_type b) const;

Return whether tensor `x` is partial contiguity between dimension `a` and dimension `b`. Partial contiguity is defined as `x.stride(i) == x.size(i + 1) * x.stride(i + 1)` for `i` between `a` and `::std::min(b - 1, x.dimension() - 1)`. If `b >= x.dimension()`, there is the additional requirement that `x.stride(x.dimension() - 1) == 1`.

As a result, `x.partialContiguity(0, x.dimension())` is equivalent to `x.isContiguous()`. However, `x.partialContiguity(0, x.dimension() - 1)` is different because it does not require that `x.stride(x.dimension() - 1) == 1`.

#### Exceptions

None.

### isUnique() const;

Return whether this tensor is the only one using the shared storage.

#### Exceptions

None.

### static isUnique(const Tensor &x) const;

Return whether tensor `x` is the only one using the shared storage.

#### Exceptions

None.

## Property modification

## Shape transformation and sub-tensor extraction

## Unitary mathematics

## Binary mathematics

## Ternary mathematics

## Sort

## Reduction

## Tensor construction

## Assignment operator

## Indexing operators

## Arithmetic operators

## Comparison operators

## Iterators

## Serialization
