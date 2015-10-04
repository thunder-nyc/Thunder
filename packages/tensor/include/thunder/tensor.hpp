/*
 * \copyright Copyright 2014 Xiang Zhang All Rights Reserved.
 * \license @{
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @}
 */

#ifndef THUNDER_TENSOR_HPP_
#define THUNDER_TENSOR_HPP_

#include "thunder/tensor/tensor.hpp"

#include <complex>

#include "thunder/storage.hpp"
#include "thunder/serializer.hpp"
#include "thunder/tensor/index_iterator.hpp"

namespace thunder {

template < typename S = DoubleStorage >
using Tensor = tensor::Tensor< S >;
typedef Tensor< SizeStorage > SizeTensor;
typedef Tensor< DoubleStorage > DoubleTensor;
typedef Tensor< FloatStorage > FloatTensor;

template < typename D = double,
           typename A = ::std::allocator< ::std::complex< D > > >
using ComplexTensor = Tensor< ComplexStorage< D, A > >;
typedef ComplexTensor< double, ::std::allocator< ::std::complex< double > > >
DoubleComplexTensor;
typedef ComplexTensor< float, ::std::allocator< ::std::complex< float > > >
FloatComplexTensor;

}  // namespace thunder


namespace thunder {
namespace tensor {

// Index iterator instantiation
extern template class IndexIterator< SizeStorage >;

// Tensor instantiation
extern template class Tensor< DoubleStorage >;
extern template class Tensor< FloatStorage >;
extern template class Tensor< DoubleComplexStorage >;
extern template class Tensor< FloatComplexStorage >;
extern template class Tensor< SizeStorage >;

#define THUNDER_TENSOR_INSTANTIATE_UNARY(S)                             \
  extern template Tensor< S > operator+(                                \
      typename Tensor< S >::const_reference value, const Tensor< S > &x); \
  extern template Tensor< S > operator-(                                \
      typename Tensor< S >::const_reference value, const Tensor< S > &x); \
  extern template Tensor< S > operator==(                                \
      typename Tensor< S >::const_reference value, const Tensor< S > &x); \
  extern template Tensor< S > operator!=(                               \
      typename Tensor< S >::const_reference value, const Tensor< S > &x); \
  extern template Tensor< S > operator>(                                \
      typename Tensor< S >::const_reference value, const Tensor< S > &x); \
  extern template Tensor< S > operator<(                                \
      typename Tensor< S >::const_reference value, const Tensor< S > &x); \
  extern template Tensor< S > operator>=(                               \
      typename Tensor< S >::const_reference value, const Tensor< S > &x); \
  extern template Tensor< S > operator<=(                               \
      typename Tensor< S >::const_reference value, const Tensor< S > &x);

#define THUNDER_TENSOR_EXPAND_UNARY(INSTANTIATE)        \
  INSTANTIATE(DoubleStorage);                           \
  INSTANTIATE(FloatStorage);                            \
  INSTANTIATE(DoubleComplexStorage);                    \
  INSTANTIATE(FloatComplexStorage);                     \
  INSTANTIATE(SizeStorage);

THUNDER_TENSOR_EXPAND_UNARY(THUNDER_TENSOR_INSTANTIATE_UNARY);

#undef THUNDER_TENSOR_INSTANTIATE_UNARY
#undef THUNDER_TENSOR_EXPAND_UNARY

#define THUNDER_TENSOR_INSTANTIATE_BINARY_INCOMPATIBLE(S1, S2)          \
  extern template Tensor< S1 >::Tensor(                                 \
      const Tensor< S2 > &y, typename Tensor< S1 >::allocator_type alloc); \
  extern template Tensor< S1 > Tensor< S1 >::viewAs(                    \
      const Tensor< S2 > &y, typename Tensor< S1 >::size_type os) const; \
  extern template Tensor< S1 > Tensor< S1 >::viewAs(                    \
      const Tensor< S2 > &y, typename Tensor< S1 >::stride_storage st,  \
      typename Tensor< S1 >::size_type os) const;                       \
  extern template Tensor< S1 > Tensor< S1 >::viewAs(                    \
      const Tensor< S1 > &x, const Tensor< S2 > &y,                     \
      typename Tensor< S1 >::size_type os);                             \
  extern template Tensor< S1 > Tensor< S1 >::viewAs(                    \
      const Tensor< S1 > &x, const Tensor< S2 > &y,                     \
      typename Tensor< S1 >::stride_storage st,                         \
      typename Tensor< S1 >::size_type os);                             \
  extern template const Tensor< S1 >& Tensor< S1 >::polar(              \
      typename Tensor< S2 >::const_reference r,                         \
      typename Tensor< S2 >::const_reference theta) const;              \
  extern template const Tensor< S1 >& Tensor< S1 >::polar(              \
      typename Tensor< S2 >::const_reference r,                         \
      const Tensor< S2 > &theta) const;                                 \
  extern template const Tensor< S1 >& Tensor< S1 >::polar(              \
      const Tensor< S2 >& r,                                            \
      typename Tensor< S2 >::const_reference theta) const;              \
  extern template const Tensor< S1 >& Tensor< S1 >::polar(              \
      const Tensor< S2 > &r, const Tensor< S2 > &theta) const;          \
  extern template Tensor< S1 >& Tensor< S1 >::polar(                    \
      typename Tensor< S2 >::const_reference r,                         \
      typename Tensor< S2 >::const_reference theta);                    \
  extern template Tensor< S1 >& Tensor< S1 >::polar(                    \
      typename Tensor< S2 >::const_reference r,                         \
      const Tensor< S2 > &theta);                                       \
  extern template Tensor< S1 >& Tensor< S1 >::polar(                    \
      const Tensor< S2 > &r,                                            \
      typename Tensor< S2 >::const_reference theta);                    \
  extern template Tensor< S1 >& Tensor< S1 >::polar(                    \
      const Tensor< S2 > &r, const Tensor< S2 > &theta);                \
  extern template Tensor< S1 > Tensor< S1 >::polar(                     \
      const Tensor< S1 > &x,                                            \
      typename Tensor< S2 >::const_reference r,                         \
      typename Tensor< S2 >::const_reference theta);                    \
  extern template Tensor< S1 > Tensor< S1 >::polar(                     \
      const Tensor< S1 > &x,                                            \
      typename Tensor< S2 >::const_reference r,                         \
      const Tensor< S2 > &theta);                                       \
  extern template Tensor< S1 > Tensor< S1 >::polar(                     \
      const Tensor< S1 > &x,                                            \
      const Tensor< S2 > &r,                                            \
      typename Tensor< S2 >::const_reference theta);                    \
  extern template Tensor< S1 > Tensor< S1 >::polar(                     \
      const Tensor< S1 > &x,                                            \
      const Tensor< S2 > &r,                                            \
      const Tensor< S2 > &theta);

#define THUNDER_TENSOR_EXPAND_BINARY_INCOMPATIBLE(INSTANTIATE)          \
  INSTANTIATE(DoubleStorage, FloatStorage);                             \
  INSTANTIATE(DoubleStorage, DoubleComplexStorage);                     \
  INSTANTIATE(DoubleStorage, FloatComplexStorage);                      \
  INSTANTIATE(FloatStorage, DoubleStorage);                             \
  INSTANTIATE(FloatStorage, DoubleComplexStorage);                      \
  INSTANTIATE(FloatStorage, FloatComplexStorage);                       \
  INSTANTIATE(DoubleComplexStorage, DoubleStorage);                     \
  INSTANTIATE(DoubleComplexStorage, FloatStorage);                      \
  INSTANTIATE(DoubleComplexStorage, FloatComplexStorage);               \
  INSTANTIATE(FloatComplexStorage, DoubleStorage);                      \
  INSTANTIATE(FloatComplexStorage, FloatStorage);                       \
  INSTANTIATE(FloatComplexStorage, DoubleComplexStorage);               \
  INSTANTIATE(DoubleStorage, SizeStorage);                              \
  INSTANTIATE(FloatStorage, SizeStorage);                               \
  INSTANTIATE(DoubleComplexStorage, SizeStorage);                       \
  INSTANTIATE(FloatComplexStorage, SizeStorage);                        \
  INSTANTIATE(SizeStorage, DoubleStorage);                              \
  INSTANTIATE(SizeStorage, FloatStorage);                               \
  INSTANTIATE(SizeStorage, DoubleComplexStorage);                       \
  INSTANTIATE(SizeStorage, FloatComplexStorage);

THUNDER_TENSOR_EXPAND_BINARY_INCOMPATIBLE(
    THUNDER_TENSOR_INSTANTIATE_BINARY_INCOMPATIBLE);

#undef THUNDER_TENSOR_INSTANTIATE_BINARY_INCOMPATIBLE
#undef THUNDER_TENSOR_EXPAND_BINARY_INCOMPATIBLE

#define THUNDER_TENSOR_INSTANTIATE_BINARY_COMPATIBLE(S1, S2)            \
  extern template bool Tensor< S1 >::isSameSizeAs(                      \
      const Tensor< S2 > &y) const;                                     \
  extern template bool Tensor< S1 >::isSameSizeAs(                      \
      const Tensor< S1 > &x, const Tensor< S2 > &y);                    \
  extern template Tensor< S1 >& Tensor< S1 >::resizeAs(const Tensor< S2 > &y); \
  extern template Tensor< S1 >& Tensor< S1 >::resizeAs(                 \
      Tensor< S1 > *x, const Tensor< S2 > &y);                          \
  extern template Tensor< S1 > Tensor< S1 >::extract(                   \
      const Tensor< S2 > &y) const;                                     \
  extern template Tensor< S1 > Tensor< S1 >::shuffle(                   \
      const Tensor< S2 > &y) const;                                     \
  extern template Tensor< S1 > Tensor< S1 >::permute(                   \
      const Tensor< S2 > &y, typename Tensor< S1 >::dim_type d) const;  \
  extern template Tensor< S2 > Tensor< S1 >::getReal() const;           \
  extern template Tensor< S2 > Tensor< S1 >::getImag() const;           \
  extern template Tensor< S2 > Tensor< S1 >::getArg() const;            \
  extern template Tensor< S2 > Tensor< S1 >::getCnrm() const;           \
  extern template Tensor< S1 > Tensor< S1 >::extract(                   \
      const Tensor< S1 > &x, const Tensor< S2 > &y);                    \
  extern template Tensor< S1 > Tensor< S1 >::shuffle(                   \
      const Tensor< S1 > &x, const Tensor< S2 > &y);                    \
  extern template Tensor< S2 > Tensor< S1 >::getReal(const Tensor< S1 > &x); \
  extern template Tensor< S2 > Tensor< S1 >::getImag(const Tensor< S1 > &x); \
  extern template Tensor< S2 > Tensor< S1 >::getArg(const Tensor< S1 > &x); \
  extern template Tensor< S2 > Tensor< S1 >::getCnrm(const Tensor< S1 > &x); \
  extern template Tensor< S2 > Tensor< S1 >::type(                      \
      typename Tensor< S2 >::allocator_type alloc) const;               \
  extern template Tensor< S2 > Tensor< S1 >::type(                      \
      const Tensor< S1 > &x,                                            \
      typename Tensor< S2 >::allocator_type alloc);                     \
  extern template const Tensor< S1 >& Tensor< S1 >::copy(               \
      const Tensor< S2 > &x) const;                                     \
  extern template Tensor< S1 >& Tensor< S1 >::copy(const Tensor< S2 > &x); \
  extern template Tensor< S1 > Tensor< S1 >::polars(                    \
      typename Tensor< S2 >::const_reference r,                         \
      const Tensor< S2 > &theta,                                        \
      typename Tensor< S1 >::allocator_type alloc);                     \
  extern template Tensor< S1 > Tensor< S1 >::polars(                    \
      const Tensor< S2 >& r,                                            \
      typename Tensor< S2 >::const_reference theta,                     \
      typename Tensor< S1 >::allocator_type alloc);                     \
  extern template Tensor< S1 > Tensor< S1 >::polars(                    \
      const Tensor< S2 > &r, const Tensor< S2 > &theta,                 \
      typename Tensor< S1 >::allocator_type alloc);                     \

#define THUNDER_TENSOR_EXPAND_BINARY_COMPATIBLE(INSTANTIATE)            \
  INSTANTIATE(DoubleStorage, DoubleStorage);                            \
  INSTANTIATE(DoubleStorage, FloatStorage);                             \
  INSTANTIATE(DoubleStorage, DoubleComplexStorage);                     \
  INSTANTIATE(DoubleStorage, FloatComplexStorage);                      \
  INSTANTIATE(FloatStorage, DoubleStorage);                             \
  INSTANTIATE(FloatStorage, FloatStorage);                              \
  INSTANTIATE(FloatStorage, DoubleComplexStorage);                      \
  INSTANTIATE(FloatStorage, FloatComplexStorage);                       \
  INSTANTIATE(DoubleComplexStorage, DoubleStorage);                     \
  INSTANTIATE(DoubleComplexStorage, FloatStorage);                      \
  INSTANTIATE(DoubleComplexStorage, DoubleComplexStorage);              \
  INSTANTIATE(DoubleComplexStorage, FloatComplexStorage);               \
  INSTANTIATE(FloatComplexStorage, DoubleStorage);                      \
  INSTANTIATE(FloatComplexStorage, FloatStorage);                       \
  INSTANTIATE(FloatComplexStorage, DoubleComplexStorage);               \
  INSTANTIATE(FloatComplexStorage, FloatComplexStorage);                \
  INSTANTIATE(SizeStorage, SizeStorage);                                \
  INSTANTIATE(DoubleStorage, SizeStorage);                              \
  INSTANTIATE(FloatStorage, SizeStorage);                               \
  INSTANTIATE(DoubleComplexStorage, SizeStorage);                       \
  INSTANTIATE(FloatComplexStorage, SizeStorage);                        \
  INSTANTIATE(SizeStorage, DoubleStorage);                              \
  INSTANTIATE(SizeStorage, FloatStorage);                               \
  INSTANTIATE(SizeStorage, DoubleComplexStorage);                       \
  INSTANTIATE(SizeStorage, FloatComplexStorage);

THUNDER_TENSOR_EXPAND_BINARY_COMPATIBLE(
    THUNDER_TENSOR_INSTANTIATE_BINARY_COMPATIBLE);

#undef THUNDER_TENSOR_INSTANTIATE_BINARY_COMPATIBLE
#undef THUNDER_TENSOR_EXPAND_BINARY_COMPATIBLE

}  // namespace tensor
}  // namespace thunder

namespace thunder {
namespace serializer {

#define THUNDER_TENSOR_INSTANTIATE_SERIALIZE(D)         \
  extern template void save(                            \
      StringBinarySerializer *s,                        \
      const ::thunder::tensor::Tensor< D > &t);         \
  extern template void StringBinarySerializer::save(    \
      const ::thunder::tensor::Tensor< D > &t);         \
  extern template void load(                            \
      StringBinarySerializer *s,                        \
      ::thunder::tensor::Tensor< D > *t);               \
  extern template void StringBinarySerializer::load(    \
      ::thunder::tensor::Tensor< D > *t);               \
  extern template void save(                            \
      FileBinarySerializer *s,                          \
      const ::thunder::tensor::Tensor< D > &t);         \
  extern template void FileBinarySerializer::save(      \
      const ::thunder::tensor::Tensor< D > &t);         \
  extern template void load(                            \
      FileBinarySerializer *s,                          \
      ::thunder::tensor::Tensor< D > *t);               \
  extern template void FileBinarySerializer::load(      \
      ::thunder::tensor::Tensor< D > *t);               \
  extern template void save(                            \
      StringTextSerializer *s,                          \
      const ::thunder::tensor::Tensor< D > &t);         \
  extern template void StringTextSerializer::save(      \
      const ::thunder::tensor::Tensor< D > &t);         \
  extern template void load(                            \
      StringTextSerializer *s,                          \
      ::thunder::tensor::Tensor< D > *t);               \
  extern template void StringTextSerializer::load(      \
      ::thunder::tensor::Tensor< D > *t);               \
  extern template void save(                            \
      FileTextSerializer *s,                            \
      const ::thunder::tensor::Tensor< D > &t);         \
  extern template void FileTextSerializer::save(        \
      const ::thunder::tensor::Tensor< D > &t);         \
  extern template void load(                            \
      FileTextSerializer *s,                            \
      ::thunder::tensor::Tensor< D > *t);               \
  extern template void FileTextSerializer::load(        \
      ::thunder::tensor::Tensor< D > *t);

#define THUNDER_TENSOR_EXPAND_SERIALIZE(INSTANTIATE)    \
  INSTANTIATE(DoubleStorage);                           \
  INSTANTIATE(FloatStorage);                            \
  INSTANTIATE(DoubleComplexStorage);                    \
  INSTANTIATE(FloatComplexStorage);                     \
  INSTANTIATE(SizeStorage);

THUNDER_TENSOR_EXPAND_SERIALIZE(THUNDER_TENSOR_INSTANTIATE_SERIALIZE);

#undef THUNDER_TENSOR_INSTANTIATE_SERIALIZE
#undef THUNDER_TENSOR_EXPAND_SERIALIZE

}  // namespace serializer
}  // namespace thunder

#endif  // THUNDER_TENSOR_HPP_
