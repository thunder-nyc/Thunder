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

#include "thunder/exception.hpp"
#include "thunder/tensor/math.hpp"

namespace thunder {
namespace tensor {

#define THUNDER_TENSOR_DEFINE_UNARY(func)                               \
  template < typename S >                                               \
  const Tensor< S >& Tensor< S >::func() const {                        \
    return math::func(*this);                                           \
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

THUNDER_TENSOR_DEFINE_UNARY(abs);
THUNDER_TENSOR_DEFINE_UNARY(fabs);
THUNDER_TENSOR_DEFINE_UNARY(exp);
THUNDER_TENSOR_DEFINE_UNARY(exp2);
THUNDER_TENSOR_DEFINE_UNARY(expm1);
THUNDER_TENSOR_DEFINE_UNARY(log);
THUNDER_TENSOR_DEFINE_UNARY(log10);
THUNDER_TENSOR_DEFINE_UNARY(log2);
THUNDER_TENSOR_DEFINE_UNARY(log1p);
THUNDER_TENSOR_DEFINE_UNARY(sqrt);
THUNDER_TENSOR_DEFINE_UNARY(cbrt);
THUNDER_TENSOR_DEFINE_UNARY(sin);
THUNDER_TENSOR_DEFINE_UNARY(cos);
THUNDER_TENSOR_DEFINE_UNARY(tan);
THUNDER_TENSOR_DEFINE_UNARY(asin);
THUNDER_TENSOR_DEFINE_UNARY(acos);
THUNDER_TENSOR_DEFINE_UNARY(atan);
THUNDER_TENSOR_DEFINE_UNARY(sinh);
THUNDER_TENSOR_DEFINE_UNARY(cosh);
THUNDER_TENSOR_DEFINE_UNARY(tanh);
THUNDER_TENSOR_DEFINE_UNARY(asinh);
THUNDER_TENSOR_DEFINE_UNARY(acosh);
THUNDER_TENSOR_DEFINE_UNARY(atanh);
THUNDER_TENSOR_DEFINE_UNARY(erf);
THUNDER_TENSOR_DEFINE_UNARY(erfc);
THUNDER_TENSOR_DEFINE_UNARY(tgamma);
THUNDER_TENSOR_DEFINE_UNARY(lgamma);
THUNDER_TENSOR_DEFINE_UNARY(ceil);
THUNDER_TENSOR_DEFINE_UNARY(floor);
THUNDER_TENSOR_DEFINE_UNARY(trunc);
THUNDER_TENSOR_DEFINE_UNARY(round);
THUNDER_TENSOR_DEFINE_UNARY(nearbyint);
THUNDER_TENSOR_DEFINE_UNARY(rint);
THUNDER_TENSOR_DEFINE_UNARY(logb);
THUNDER_TENSOR_DEFINE_UNARY(fpclassify);
THUNDER_TENSOR_DEFINE_UNARY(isfinite);
THUNDER_TENSOR_DEFINE_UNARY(isinf);
THUNDER_TENSOR_DEFINE_UNARY(isnan);
THUNDER_TENSOR_DEFINE_UNARY(isnormal);
THUNDER_TENSOR_DEFINE_UNARY(signbit);
THUNDER_TENSOR_DEFINE_UNARY(zero);
THUNDER_TENSOR_DEFINE_UNARY(real);
THUNDER_TENSOR_DEFINE_UNARY(imag);
THUNDER_TENSOR_DEFINE_UNARY(arg);
THUNDER_TENSOR_DEFINE_UNARY(cnrm);
THUNDER_TENSOR_DEFINE_UNARY(conj);
THUNDER_TENSOR_DEFINE_UNARY(proj);

#undef THUNDER_TENSOR_DEFINE_UNARY

}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_INL_UNARY_HPP_
