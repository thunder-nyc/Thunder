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

#include "thunder/tensor/tensor.hpp"
#include "thunder/tensor/tensor-inl.hpp"

#include <complex>
#include <utility>

#include "thunder/storage.hpp"
#include "thunder/storage/storage.hpp"
#include "thunder/storage/storage-inl.hpp"

namespace thunder {
namespace storage {

// Existing specializations
extern template class Storage< double >;
extern template class Storage< float >;
extern template class Storage< ::std::complex< double > >;
extern template class Storage< ::std::complex< float > >;

// Specialization of additional storage class
template class Storage< ::std::size_t >;
template class Storage< ::std::ptrdiff_t >;
template class Storage< ::std::pair< ::std::size_t, ::std::size_t > >;

}  // namespace storage
}  // namespace thunder

namespace thunder {
namespace tensor {

// Tensor instantiation
template class Tensor< DoubleStorage >;
template class Tensor< FloatStorage >;
template class Tensor< DoubleComplexStorage >;
template class Tensor< FloatComplexStorage >;

#define THUNDER_TENSOR_INSTANTIATE_UNARY(S)                             \
  template Tensor< S > operator+(                                       \
      typename Tensor< S >::const_reference value, const Tensor< S > &x); \
  template Tensor< S > operator-(                                       \
      typename Tensor< S >::const_reference value, const Tensor< S > &x); \
  template Tensor< S > operator==(                                      \
      typename Tensor< S >::const_reference value, const Tensor< S > &x); \
  template Tensor< S > operator!=(                                      \
      typename Tensor< S >::const_reference value, const Tensor< S > &x); \
  template Tensor< S > operator>(                                       \
      typename Tensor< S >::const_reference value, const Tensor< S > &x); \
  template Tensor< S > operator<(                                       \
      typename Tensor< S >::const_reference value, const Tensor< S > &x); \
  template Tensor< S > operator>=(                                      \
      typename Tensor< S >::const_reference value, const Tensor< S > &x); \
  template Tensor< S > operator<=(                                      \
      typename Tensor< S >::const_reference value, const Tensor< S > &x);

#define THUNDER_TENSOR_EXPAND_UNARY(INSTANTIATE)        \
  INSTANTIATE(DoubleStorage);                           \
  INSTANTIATE(FloatStorage);                            \
  INSTANTIATE(DoubleComplexStorage);                    \
  INSTANTIATE(FloatComplexStorage);

THUNDER_TENSOR_EXPAND_UNARY(THUNDER_TENSOR_INSTANTIATE_UNARY);

#undef THUNDER_TENSOR_INSTANTIATE_UNARY
#undef THUNDER_TENSOR_EXPAND_UNARY

#define THUNDER_TENSOR_INSTANTIATE_BINARY_INCOMPATIBLE(S1, S2)          \
  template Tensor< S1 >::Tensor(const Tensor< S2 > &y);                 \
  template Tensor< S1 > Tensor< S1 >::viewAs(                           \
      const Tensor< S2 > &y, typename Tensor< S1 >::size_type os) const;\
  template Tensor< S1 > Tensor< S1 >::viewAs(                           \
      const Tensor< S2 > &y, typename Tensor< S1 >::stride_storage st,  \
      typename Tensor< S1 >::size_type os) const;                       \
  template Tensor< S1 > Tensor< S1 >::viewAs(                           \
      const Tensor< S1 > &x, const Tensor< S2 > &y,                     \
      typename Tensor< S1 >::size_type os);                             \
  template Tensor< S1 > Tensor< S1 >::viewAs(                           \
      const Tensor< S1 > &x, const Tensor< S2 > &y,                     \
      typename Tensor< S1 >::stride_storage st,                         \
      typename Tensor< S1 >::size_type os);                             \
  template const Tensor< S1 >& Tensor< S1 >::polar(                     \
      typename Tensor< S2 >::const_reference r,                         \
      typename Tensor< S2 >::const_reference theta) const;              \
  template const Tensor< S1 >& Tensor< S1 >::polar(                     \
      typename Tensor< S2 >::const_reference r,                         \
      const Tensor< S2 > &theta) const;                                 \
  template const Tensor< S1 >& Tensor< S1 >::polar(                     \
      const Tensor< S2 >& r,                                            \
      typename Tensor< S2 >::const_reference theta) const;              \
  template const Tensor< S1 >& Tensor< S1 >::polar(                     \
      const Tensor< S2 > &r, const Tensor< S2 > &theta) const;          \
  template Tensor< S1 >& Tensor< S1 >::polar(                           \
      typename Tensor< S2 >::const_reference r,                         \
      typename Tensor< S2 >::const_reference theta);                    \
  template Tensor< S1 >& Tensor< S1 >::polar(                           \
      typename Tensor< S2 >::const_reference r,                         \
      const Tensor< S2 > &theta);                                       \
  template Tensor< S1 >& Tensor< S1 >::polar(                           \
      const Tensor< S2 > &r,                                            \
      typename Tensor< S2 >::const_reference theta);                    \
  template Tensor< S1 >& Tensor< S1 >::polar(                           \
      const Tensor< S2 > &r, const Tensor< S2 > &theta);                \
  template Tensor< S1 > Tensor< S1 >::polar(                            \
      const Tensor< S1 > &x,                                            \
      typename Tensor< S2 >::const_reference r,                         \
      typename Tensor< S2 >::const_reference theta);                    \
  template Tensor< S1 > Tensor< S1 >::polar(                            \
      const Tensor< S1 > &x,                                            \
      typename Tensor< S2 >::const_reference r,                         \
      const Tensor< S2 > &theta);                                       \
  template Tensor< S1 > Tensor< S1 >::polar(                            \
      const Tensor< S1 > &x,                                            \
      const Tensor< S2 > &r,                                            \
      typename Tensor< S2 >::const_reference theta);                    \
  template Tensor< S1 > Tensor< S1 >::polar(                            \
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

THUNDER_TENSOR_EXPAND_BINARY_INCOMPATIBLE(
    THUNDER_TENSOR_INSTANTIATE_BINARY_INCOMPATIBLE);

#undef THUNDER_TENSOR_INSTANTIATE_BINARY_INCOMPATIBLE
#undef THUNDER_TENSOR_EXPAND_BINARY_INCOMPATIBLE

#define THUNDER_TENSOR_INSTANTIATE_BINARY_COMPATIBLE(S1, S2)            \
  template bool Tensor< S1 >::isSameSizeAs(const Tensor< S2 > &y) const; \
  template bool Tensor< S1 >::isSameSizeAs(                             \
      const Tensor< S1 > &x, const Tensor< S2 > &y);                    \
  template Tensor< S1 >& Tensor< S1 >::resizeAs(const Tensor< S2 > &y); \
  template Tensor< S1 >& Tensor< S1 >::resizeAs(                        \
      Tensor< S1 > *x, const Tensor< S2 > &y);                          \
  template Tensor< S1 > Tensor< S1 >::extract(                          \
      const Tensor< S2 > &y) const;                                     \
  template Tensor< S1 > Tensor< S1 >::shuffle(                          \
      const Tensor< S2 > &y) const;                                     \
  template Tensor< S2 > Tensor< S1 >::getReal() const;                  \
  template Tensor< S2 > Tensor< S1 >::getImag() const;                  \
  template Tensor< S2 > Tensor< S1 >::getArg() const;                   \
  template Tensor< S2 > Tensor< S1 >::getCnrm() const;                  \
  template Tensor< S1 > Tensor< S1 >::extract(                          \
      const Tensor< S1 > &x, const Tensor< S2 > &y);                    \
  template Tensor< S1 > Tensor< S1 >::shuffle(                          \
      const Tensor< S1 > &x, const Tensor< S2 > &y);                    \
  template Tensor< S2 > Tensor< S1 >::getReal(const Tensor< S1 > &x);   \
  template Tensor< S2 > Tensor< S1 >::getImag(const Tensor< S1 > &x);   \
  template Tensor< S2 > Tensor< S1 >::getArg(const Tensor< S1 > &x);    \
  template Tensor< S2 > Tensor< S1 >::getCnrm(const Tensor< S1 > &x);   \
  template Tensor< S2 > Tensor< S1 >::type() const;                     \
  template Tensor< S2 > Tensor< S1 >::type(const Tensor< S1 > &x);      \
  template const Tensor< S1 >& Tensor< S1 >::copy(                      \
      const Tensor< S2 > &x) const;                                     \
  template Tensor< S1 >& Tensor< S1 >::copy(const Tensor< S2 > &x);     \
  template Tensor< S1 > Tensor< S1 >::polars(                           \
      typename Tensor< S2 >::const_reference r,                         \
      const Tensor< S2 > &theta);                                       \
  template Tensor< S1 > Tensor< S1 >::polars(                           \
      const Tensor< S2 >& r,                                            \
      typename Tensor< S2 >::const_reference theta);                    \
  template Tensor< S1 > Tensor< S1 >::polars(                           \
      const Tensor< S2 > &r, const Tensor< S2 > &theta);

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

THUNDER_TENSOR_EXPAND_BINARY_COMPATIBLE(
    THUNDER_TENSOR_INSTANTIATE_BINARY_COMPATIBLE);

#undef THUNDER_TENSOR_INSTANTIATE_BINARY_COMPATIBLE
#undef THUNDER_TENSOR_EXPAND_BINARY_COMPATIBLE

}  // namespace tensor
}  // namespace thunder
