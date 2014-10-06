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

#ifndef THUNDER_TENSOR_COMPLEX_HPP_
#define THUNDER_TENSOR_COMPLEX_HPP_

#include <complex>

#include "thunder/tensor/tensor.hpp"
#include "thunder/tensor/math.hpp"

namespace thunder {
namespace tensor {
namespace math {

// Unary operations
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& erf(
    const Tensor< Storage < ::std::complex< D >, A > > &x);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& erfc(
    const Tensor< Storage < ::std::complex< D >, A > > &x);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& tgamma(
    const Tensor< Storage < ::std::complex< D >, A > > &x);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& lgamma(
    const Tensor< Storage < ::std::complex< D >, A > > &x);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& ceil(
    const Tensor< Storage < ::std::complex< D >, A > > &x);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& floor(
    const Tensor< Storage < ::std::complex< D >, A > > &x);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& trunc(
    const Tensor< Storage < ::std::complex< D >, A > > &x);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& round(
    const Tensor< Storage < ::std::complex< D >, A > > &x);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& nearbyint(
    const Tensor< Storage < ::std::complex< D >, A > > &x);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& rint(
    const Tensor< Storage < ::std::complex< D >, A > > &x);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& fpclassify(
    const Tensor< Storage < ::std::complex< D >, A > > &x);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& isfinite(
    const Tensor< Storage < ::std::complex< D >, A > > &x);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& isinf(
    const Tensor< Storage < ::std::complex< D >, A > > &x);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& isnan(
    const Tensor< Storage < ::std::complex< D >, A > > &x);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& isnormal(
    const Tensor< Storage < ::std::complex< D >, A > > &x);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& signbit(
    const Tensor< Storage < ::std::complex< D >, A > > &x);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& fabs(
    const Tensor< Storage < ::std::complex< D >, A > > &x);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& exp2(
    const Tensor< Storage < ::std::complex< D >, A > > &x);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& expm1(
    const Tensor< Storage < ::std::complex< D >, A > > &x);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& log1p(
    const Tensor< Storage < ::std::complex< D >, A > > &x);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& cbrt(
    const Tensor< Storage < ::std::complex< D >, A > > &x);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& log2(
    const Tensor< Storage < ::std::complex< D >, A > > &x);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& logb(
    const Tensor< Storage < ::std::complex< D >, A > > &x);

// Binary operations with a value
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& fmod(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& remainder(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& fmax(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& fmin(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& fdim(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& hypot(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& atan2(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& ldexp(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& scalbn(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& scalbln(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& nextafter(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& nexttoward(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& copysign(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& isgreater(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& isgreaterequal(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& isless(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& islessequal(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& islessgreater(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& isunordered(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y);

// Binary operations with another tensor
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& fmod(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& remainder(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& fmax(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& fmin(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& fdim(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& hypot(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& atan2(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& ldexp(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& scalbn(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& scalbln(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& nextafter(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& nexttoward(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& copysign(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& isgreater(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& isgreaterequal(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& isless(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& islessequal(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& islessgreater(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& isunordered(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y);

// Ternary functions
template < typename D, typename A, typename T2 >
const Tensor< Storage< ::std::complex< D >, A > >& polar(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const T2 &y, typename T2::const_reference z);
template < typename D, typename A, typename T2 >
const Tensor< Storage< ::std::complex< D >, A > >& polar(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename T2::const_reference y, const T2 &z);
template < typename D, typename A, typename S >
const Tensor< Storage< ::std::complex< D >, A > >& polar(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< S > &y, const Tensor< S > &z);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& polar(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference z);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& polar(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference z);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& polar(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y,
    const Tensor< Storage< ::std::complex< D >, A > > &z);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& polar(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y,
    const Tensor< Storage< ::std::complex< D >, A > > &z);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& fma(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference z);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& fma(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference z);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& fma(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y,
    const Tensor< Storage< ::std::complex< D >, A > > &z);
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& fma(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y,
    const Tensor< Storage< ::std::complex< D >, A > > &z);

// Reduction functions to a single value
template < typename D, typename A >
typename Tensor< Storage< ::std::complex< D >, A > >::value_type max(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    Tensor< typename Tensor< Storage< ::std::complex< D >, A > >
    ::size_storage> *pos);
template < typename D, typename A >
typename Tensor< Storage< ::std::complex< D >, A > >::value_type min(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    Tensor< typename Tensor< Storage< ::std::complex< D >, A > >
    ::size_storage> *pos);
template < typename D, typename A >
typename Tensor< Storage< ::std::complex< D >, A > >::value_type max(
    const Tensor< Storage< ::std::complex< D >, A > > &x);
template < typename D, typename A >
typename Tensor< Storage< ::std::complex< D >, A > >::value_type min(
    const Tensor< Storage< ::std::complex< D >, A > > &x);

// Reduction functions along a dimension
template < typename D, typename A >
Tensor< Storage< ::std::complex< D >, A > > max(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::dim_type d,
    Tensor< typename Tensor< Storage< ::std::complex< D >, A > >
    ::size_storage> *pos);
template < typename D, typename A >
Tensor< Storage< ::std::complex< D >, A > > min(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::dim_type d,
    Tensor< typename Tensor< Storage< ::std::complex< D >, A > >
    ::size_storage> *pos);
template < typename D, typename A >
Tensor< Storage< ::std::complex< D >, A > > max(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::dim_type d);
template < typename D, typename A >
Tensor< Storage< ::std::complex< D >, A > > min(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::dim_type d);

}  // namespace math
}  // namespace tensor
}  // namespace thunder

#include "thunder/tensor/complex-inl.hpp"

#endif  // THUNDER_TENSOR_COMPLEX_HPP_
