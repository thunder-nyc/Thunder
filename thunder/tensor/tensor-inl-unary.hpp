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

#include "thunder/exception.hpp"

namespace thunder {
namespace tensor {

#define THUNDER_TENSOR_DECLARE_UNARY(func)                              \
  template < typename S >                                               \
  const Tensor< S >& Tensor< S >::func() const {                        \
    if (partialContiguity(0, size_.size() - 1)) {                       \
      pointer data_pointer = data();                                    \
      difference_type data_step = stride_[stride_.size() - 1];          \
      size_type data_length = length();                                 \
      for (size_type i = 0; i < data_length; ++i) {                     \
        data_pointer[i * step] = ::std::func(data_pointer[i * step]);   \
      }                                                                 \
    } else {                                                            \
      for (reference_iterator begin = reference_begin(),                \
               end = reference_end(); begin != end; ++begin) {          \
        *begin = ::std::func(*begin);                                   \
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

THUNDER_TENSOR_DECLARE_UNARY(abs);
THUNDER_TENSOR_DECLARE_UNARY(exp);
THUNDER_TENSOR_DECLARE_UNARY(exp2);
THUNDER_TENSOR_DECLARE_UNARY(expm1);
THUNDER_TENSOR_DECLARE_UNARY(log);
THUNDER_TENSOR_DECLARE_UNARY(log10);
THUNDER_TENSOR_DECLARE_UNARY(log2);
THUNDER_TENSOR_DECLARE_UNARY(log1p);
THUNDER_TENSOR_DECLARE_UNARY(sqrt);
THUNDER_TENSOR_DECLARE_UNARY(cbrt);
THUNDER_TENSOR_DECLARE_UNARY(sin);
THUNDER_TENSOR_DECLARE_UNARY(cos);
THUNDER_TENSOR_DECLARE_UNARY(tan);
THUNDER_TENSOR_DECLARE_UNARY(asin);
THUNDER_TENSOR_DECLARE_UNARY(acos);
THUNDER_TENSOR_DECLARE_UNARY(atan);
THUNDER_TENSOR_DECLARE_UNARY(sinh);
THUNDER_TENSOR_DECLARE_UNARY(cosh);
THUNDER_TENSOR_DECLARE_UNARY(tanh);
THUNDER_TENSOR_DECLARE_UNARY(asinh);
THUNDER_TENSOR_DECLARE_UNARY(acosh);
THUNDER_TENSOR_DECLARE_UNARY(atanh);
THUNDER_TENSOR_DECLARE_UNARY(erf);
THUNDER_TENSOR_DECLARE_UNARY(erfc);
THUNDER_TENSOR_DECLARE_UNARY(tgamma);
THUNDER_TENSOR_DECLARE_UNARY(lgamma);
THUNDER_TENSOR_DECLARE_UNARY(ceil);
THUNDER_TENSOR_DECLARE_UNARY(floor);
THUNDER_TENSOR_DECLARE_UNARY(trunc);
THUNDER_TENSOR_DECLARE_UNARY(round);
THUNDER_TENSOR_DECLARE_UNARY(nearbyint);
THUNDER_TENSOR_DECLARE_UNARY(rint);
THUNDER_TENSOR_DECLARE_UNARY(logb);
THUNDER_TENSOR_DECLARE_UNARY(fpclassify);
THUNDER_TENSOR_DECLARE_UNARY(isfinite);
THUNDER_TENSOR_DECLARE_UNARY(isinf);
THUNDER_TENSOR_DECLARE_UNARY(isnan);
THUNDER_TENSOR_DECLARE_UNARY(isnormal);
THUNDER_TENSOR_DECLARE_UNARY(signbit);
THUNDER_TENSOR_DECLARE_UNARY(zero);
THUNDER_TENSOR_DECLARE_UNARY(real);
THUNDER_TENSOR_DECLARE_UNARY(imag);
THUNDER_TENSOR_DECLARE_UNARY(arg);
THUNDER_TENSOR_DECLARE_UNARY(cnrm);
THUNDER_TENSOR_DECLARE_UNARY(conj);
THUNDER_TENSOR_DECLARE_UNARY(proj);

#undef THUNDER_TENSOR_DECLARE_UNARY

}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_INL_UNARY_HPP_
