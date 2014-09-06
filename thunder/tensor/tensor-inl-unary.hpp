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

#ifndef THUNDER_TENSOR_TENSOR_INL_UNARY_HPP_
#define THUNDER_TENSOR_TENSOR_INL_UNARY_HPP_

#include "thunder/tensor/tensor.hpp"
#include "thunder/tensor/tensor-inl.hpp"

#include <cmath>
#include <complex>

#include "thunder/exception.hpp"

namespace thunder {
namespace tensor {

#define THUNDER_TENSOR_DECLARE_STD_UNARY(func)                          \
  template < typename S >                                               \
  const Tensor< S >& Tensor< S >::func() const {                        \
    if (partialContiguity(0, size_.size() - 1)) {                       \
      pointer data_pointer = data();                                    \
      difference_type data_step = stride_[stride_.size() - 1];          \
      size_type data_length = length();                                 \
      for (size_type i = 0; i < data_length; ++i) {                     \
        data_pointer[i * data_step] = static_cast< value_type >(        \
            ::std::func(data_pointer[i * data_step]));                  \
      }                                                                 \
    } else {                                                            \
      for (reference_iterator begin = reference_begin(),                \
               end = reference_end(); begin != end; ++begin) {          \
        *begin = static_cast< value_type >(::std::func(*begin));        \
      }                                                                 \
    }                                                                   \
    return *this;                                                       \
  }                                                                     \
  template < typename S >                                               \
  Tensor< S >& Tensor< S >::func() {                                    \
    return const_cast< Tensor& >(                                       \
        const_cast< const Tensor* >(this)->func());                     \
  }                                                                     \
  template < typename S >                                               \
  Tensor< S > Tensor< S >::func(const Tensor &x) {                      \
    return x.clone().func();                                            \
  }

THUNDER_TENSOR_DECLARE_STD_UNARY(abs);
THUNDER_TENSOR_DECLARE_STD_UNARY(exp);
THUNDER_TENSOR_DECLARE_STD_UNARY(exp2);
THUNDER_TENSOR_DECLARE_STD_UNARY(expm1);
THUNDER_TENSOR_DECLARE_STD_UNARY(log);
THUNDER_TENSOR_DECLARE_STD_UNARY(log10);
THUNDER_TENSOR_DECLARE_STD_UNARY(log2);
THUNDER_TENSOR_DECLARE_STD_UNARY(log1p);
THUNDER_TENSOR_DECLARE_STD_UNARY(sqrt);
THUNDER_TENSOR_DECLARE_STD_UNARY(cbrt);
THUNDER_TENSOR_DECLARE_STD_UNARY(sin);
THUNDER_TENSOR_DECLARE_STD_UNARY(cos);
THUNDER_TENSOR_DECLARE_STD_UNARY(tan);
THUNDER_TENSOR_DECLARE_STD_UNARY(asin);
THUNDER_TENSOR_DECLARE_STD_UNARY(acos);
THUNDER_TENSOR_DECLARE_STD_UNARY(atan);
THUNDER_TENSOR_DECLARE_STD_UNARY(sinh);
THUNDER_TENSOR_DECLARE_STD_UNARY(cosh);
THUNDER_TENSOR_DECLARE_STD_UNARY(tanh);
THUNDER_TENSOR_DECLARE_STD_UNARY(asinh);
THUNDER_TENSOR_DECLARE_STD_UNARY(acosh);
THUNDER_TENSOR_DECLARE_STD_UNARY(atanh);
THUNDER_TENSOR_DECLARE_STD_UNARY(erf);
THUNDER_TENSOR_DECLARE_STD_UNARY(erfc);
THUNDER_TENSOR_DECLARE_STD_UNARY(tgamma);
THUNDER_TENSOR_DECLARE_STD_UNARY(lgamma);
THUNDER_TENSOR_DECLARE_STD_UNARY(ceil);
THUNDER_TENSOR_DECLARE_STD_UNARY(floor);
THUNDER_TENSOR_DECLARE_STD_UNARY(trunc);
THUNDER_TENSOR_DECLARE_STD_UNARY(round);
THUNDER_TENSOR_DECLARE_STD_UNARY(nearbyint);
THUNDER_TENSOR_DECLARE_STD_UNARY(rint);
THUNDER_TENSOR_DECLARE_STD_UNARY(logb);
THUNDER_TENSOR_DECLARE_STD_UNARY(fpclassify);
THUNDER_TENSOR_DECLARE_STD_UNARY(isfinite);
THUNDER_TENSOR_DECLARE_STD_UNARY(isinf);
THUNDER_TENSOR_DECLARE_STD_UNARY(isnan);
THUNDER_TENSOR_DECLARE_STD_UNARY(isnormal);
THUNDER_TENSOR_DECLARE_STD_UNARY(signbit);
THUNDER_TENSOR_DECLARE_STD_UNARY(real);
THUNDER_TENSOR_DECLARE_STD_UNARY(imag);
THUNDER_TENSOR_DECLARE_STD_UNARY(arg);
THUNDER_TENSOR_DECLARE_STD_UNARY(conj);
THUNDER_TENSOR_DECLARE_STD_UNARY(proj);

#undef THUNDER_TENSOR_DECLARE_STD_UNARY

template < typename S >
const Tensor< S >& Tensor< S >::cnrm() const {
  if (partialContiguity(0, size_.size() - 1)) {
    pointer data_pointer = data();
    difference_type data_step = stride_[stride_.size() - 1];
    size_type data_length = length();
    for (size_type i = 0; i < data_length; ++i) {
      data_pointer[i * data_step] = static_cast< value_type >(
          ::std::norm(data_pointer[i * data_step]));
    }
  } else {
    for (reference_iterator begin = reference_begin(),
             end = reference_end(); begin != end; ++begin) {
      *begin = static_cast< value_type >(::std::norm(*begin));
    }
  }
  return *this;
}
template < typename S >
Tensor< S >& Tensor< S >::cnrm() {
  return const_cast< Tensor& >(const_cast< const Tensor* >(this)->cnrm());
}
template < typename S >
Tensor< S > Tensor< S >::cnrm(const Tensor &x) {
  return x.clone().cnrm();
}

template < typename S >
const Tensor< S >& Tensor< S >::zero() const {
  if (partialContiguity(0, size_.size() - 1)) {
    pointer data_pointer = data();
    difference_type data_step = stride_[stride_.size() - 1];
    size_type data_length = length();
    for (size_type i = 0; i < data_length; ++i) {
      data_pointer[i * data_step] = 0;
    }
  } else {
    for (reference_iterator begin = reference_begin(),
             end = reference_end(); begin != end; ++begin) {
      *begin = 0;
    }
  }
  return *this;
}
template < typename S >
Tensor< S >& Tensor< S >::zero() {
  return const_cast< Tensor& >(const_cast< const Tensor* >(this)->zero());
}
template < typename S >
Tensor< S > Tensor< S >::zero(const Tensor &x) {
  return x.clone().zero();
}

}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_INL_UNARY_HPP_
