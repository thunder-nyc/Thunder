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

#ifndef THUNDER_TENSOR_MATH_HPP_
#define THUNDER_TENSOR_MATH_HPP_

#include "thunder/tensor/tensor.hpp"

namespace thunder {
namespace tensor {
namespace math {

// Unary operations
template < typename T >
const T& abs(const T &x);
template < typename T >
const T& fabs(const T &x);
template < typename T >
const T& exp(const T &x);
template < typename T >
const T& exp2(const T &x);
template < typename T >
const T& expm1(const T &x);
template < typename T >
const T& log(const T &x);
template < typename T >
const T& log10(const T &x);
template < typename T >
const T& log2(const T &x);
template < typename T >
const T& log1p(const T &x);
template < typename T >
const T& sqrt(const T &x);
template < typename T >
const T& cbrt(const T &x);
template < typename T >
const T& sin(const T &x);
template < typename T >
const T& cos(const T &x);
template < typename T >
const T& tan(const T &x);
template < typename T >
const T& asin(const T &x);
template < typename T >
const T& acos(const T &x);
template < typename T >
const T& atan(const T &x);
template < typename T >
const T& sinh(const T &x);
template < typename T >
const T& cosh(const T &x);
template < typename T >
const T& tanh(const T &x);
template < typename T >
const T& asinh(const T &x);
template < typename T >
const T& acosh(const T &x);
template < typename T >
const T& atanh(const T &x);
template < typename T >
const T& erf(const T &x);
template < typename T >
const T& erfc(const T &x);
template < typename T >
const T& tgamma(const T &x);
template < typename T >
const T& lgamma(const T &x);
template < typename T >
const T& ceil(const T &x);
template < typename T >
const T& floor(const T &x);
template < typename T >
const T& trunc(const T &x);
template < typename T >
const T& round(const T &x);
template < typename T >
const T& nearbyint(const T &x);
template < typename T >
const T& rint(const T &x);
template < typename T >
const T& logb(const T &x);
template < typename T >
const T& fpclassify(const T &x);
template < typename T >
const T& isfinite(const T &x);
template < typename T >
const T& isinf(const T &x);
template < typename T >
const T& isnan(const T &x);
template < typename T >
const T& isnormal(const T &x);
template < typename T >
const T& signbit(const T &x);
template < typename T >
const T& zero(const T &x);
template < typename T >
const T& real(const T &x);
template < typename T >
const T& imag(const T &x);
template < typename T >
const T& arg(const T &x);
template < typename T >
const T& cnrm(const T &x);
template < typename T >
const T& conj(const T &x);
template < typename T >
const T& proj(const T &x);

// Element-wise operations with a value
template < typename T >
const T& add(const T &x, typename T::const_reference y);
template < typename T >
const T& sub(const T &x, typename T::const_reference y);
template < typename T >
const T& mul(const T &x, typename T::const_reference y);
template < typename T >
const T& div(const T &x, typename T::const_reference y);
template < typename T >
const T& fmod(const T &x, typename T::const_reference y);
template < typename T >
const T& remainder(const T &x, typename T::const_reference y);
template < typename T >
const T& fmax(const T &x, typename T::const_reference y);
template < typename T >
const T& fmin(const T &x, typename T::const_reference y);
template < typename T >
const T& fdim(const T &x, typename T::const_reference y);
template < typename T >
const T& pow(const T &x, typename T::const_reference y);
template < typename T >
const T& hypot(const T &x, typename T::const_reference y);
template < typename T >
const T& atan2(const T &x, typename T::const_reference y);
template < typename T >
const T& ldexp(const T &x, typename T::const_reference y);
template < typename T >
const T& scalbn(const T &x, typename T::const_reference y);
template < typename T >
const T& scalbln(const T &x, typename T::const_reference y);
template < typename T >
const T& nextafter(const T &x, typename T::const_reference y);
template < typename T >
const T& nexttoward(const T &x, typename T::const_reference y);
template < typename T >
const T& copysign(const T &x, typename T::const_reference y);
template < typename T >
const T& isgreater(const T &x, typename T::const_reference y);
template < typename T >
const T& isgreaterequal(const T &x, typename T::const_reference y);
template < typename T >
const T& isless(const T &x, typename T::const_reference y);
template < typename T >
const T& islessequal(const T &x, typename T::const_reference y);
template < typename T >
const T& islessgreater(const T &x, typename T::const_reference y);
template < typename T >
const T& isunordered(const T &x, typename T::const_reference y);
template < typename T >
const T& fill(const T &x, typename T::const_reference y);

// Element-wise operations with another tensor
template < typename T >
const T& add(const T &x, const T &y);
template < typename T >
const T& sub(const T &x, const T &y);
template < typename T >
const T& mul(const T &x, const T &y);
template < typename T >
const T& div(const T &x, const T &y);
template < typename T >
const T& fmod(const T &x, const T &y);
template < typename T >
const T& remainder(const T &x, const T &y);
template < typename T >
const T& fmax(const T &x, const T &y);
template < typename T >
const T& fmin(const T &x, const T &y);
template < typename T >
const T& fdim(const T &x, const T &y);
template < typename T >
const T& pow(const T &x, const T &y);
template < typename T >
const T& hypot(const T &x, const T &y);
template < typename T >
const T& atan2(const T &x, const T &y);
template < typename T >
const T& ldexp(const T &x, const T &y);
template < typename T >
const T& scalbn(const T &x, const T &y);
template < typename T >
const T& scalbln(const T &x, const T &y);
template < typename T >
const T& nextafter(const T &x, const T &y);
template < typename T >
const T& nexttoward(const T &x, const T &y);
template < typename T >
const T& copysign(const T &x, const T &y);
template < typename T >
const T& isgreater(const T &x, const T &y);
template < typename T >
const T& isgreaterequal(const T &x, const T &y);
template < typename T >
const T& isless(const T &x, const T &y);
template < typename T >
const T& islessequal(const T &x, const T &y);
template < typename T >
const T& islessgreater(const T &x, const T &y);
template < typename T >
const T& isunordered(const T &x, const T &y);

// Ternary functions
template < typename T >
const T& fma(
    const T &x, typename T::const_reference y, typename T::const_reference z);
template < typename T >
const T& fma(const T &x, const T &y, typename T::const_reference z);
template < typename T >
const T& fma(const T &x, typename T::const_reference y, const T &z);
template < typename T >
const T& fma(const T &x, const T &y, const T &z);

// Reduction functions to a single value
template < typename T >
const typename T::value_type max(
    const T &x, Tensor< typename T::size_storage > *pos);
template < typename T >
const typename T::value_type min(
    const T &x, Tensor< typename T::size_storage > *pos);
template < typename T >
const typename T::value_type max(const T &x);
template < typename T >
const typename T::value_type min(const T &x);
template < typename T >
const typename T::value_type sum(const T &x);
template < typename T >
const typename T::value_type prod(const T &x);
template < typename T >
const typename T::value_type mean(const T &x);
template < typename T >
const typename T::value_type var(const T &x);
template < typename T >
const typename T::value_type std(const T &x);

// Reduction functions along a particular dimension
template < typename T >
T max(const T &x, typename T::dim_type d,
      Tensor< typename T::size_storage > *pos);
template < typename T >
T min(const T &x, typename T::dim_type d,
      Tensor< typename T::size_storage > *pos);
template < typename T >
T max(const T &x, typename T::dim_type d);
template < typename T >
T min(const T &x, typename T::dim_type d);
template < typename T >
T sum(const T &x, typename T::dim_type d);
template < typename T >
T prod(const T &x, typename T::dim_type d);
template < typename T >
T mean(const T &x, typename T::dim_type d);
template < typename T >
T var(const T &x, typename T::dim_type d);
template < typename T >
T std(const T &x, typename T::dim_type d);

}  // namespace math
}  // namespace tensor
}  // namespace thunder

#include "thunder/tensor/math-inl.hpp"

#endif  // THUNDER_TENSOR_MATH_HPP_
