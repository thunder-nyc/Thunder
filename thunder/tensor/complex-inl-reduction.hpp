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

#ifndef THUNDER_TENSOR_COMPLEX_INL_REDUCTION_HPP_
#define THUNDER_TENSOR_COMPLEX_INL_REDUCTION_HPP_

#include "thunder/tensor/complex.hpp"
#include "thunder/tensor/complex-inl.hpp"

#include <cmath>
#include <complex>

#include "thunder/tensor/math.hpp"

namespace thunder {
namespace tensor {
namespace math {

template < typename D, typename A >
typename Tensor< Storage< ::std::complex< D >, A > >::value_type max(
    const Tensor< Storage< ::std::complex< D >, A > > &x) {
  throw domain_error("max is undefined for complex numbers");
  return 0;
}

template < typename D, typename A >
typename Tensor< Storage< ::std::complex< D >, A > >::value_type min(
    const Tensor< Storage< ::std::complex< D >, A > > &x) {
  throw domain_error("min is undefined for complex numbers");
  return 0;
}

template < typename D, typename A >
Tensor< Storage< ::std::complex< D >, A > > max(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::dim_type d) {
  throw domain_error("max is undefined for complex numbers");
  return Tensor< Storage< ::std::complex< D >, A > >();
}

template < typename D, typename A >
Tensor< Storage< ::std::complex< D >, A > > min(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::dim_type d) {
  throw domain_error("min is undefined for complex numbers");
  return Tensor< Storage< ::std::complex< D >, A > >();
}


template < typename D, typename A >
typename Tensor< Storage< ::std::complex< D >, A > >::value_type max(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    Tensor< typename Tensor< Storage< ::std::complex< D >, A > >
    ::size_storage> *pos) {
  throw domain_error("max is undefined for complex numbers");
  return 0;
}

template < typename D, typename A >
typename Tensor< Storage< ::std::complex< D >, A > >::value_type min(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    Tensor< typename Tensor< Storage< ::std::complex< D >, A > >
    ::size_storage> *pos) {
  throw domain_error("max is undefined for complex numbers");
  return 0;
}

template < typename D, typename A >
Tensor< Storage< ::std::complex< D >, A > > max(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::dim_type d,
    Tensor< typename Tensor< Storage< ::std::complex< D >, A > >
    ::size_storage> *pos) {
  throw domain_error("max is undefined for complex numbers");
  return Tensor< Storage< ::std::complex< D >, A > >();
}

template < typename D, typename A >
Tensor< Storage< ::std::complex< D >, A > > min(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::dim_type d,
    Tensor< typename Tensor< Storage< ::std::complex< D >, A > >
    ::size_storage> *pos) {
  throw domain_error("max is undefined for complex numbers");
  return Tensor< Storage< ::std::complex< D >, A > >();
}

}  // namespace math
}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_COMPLEX_INL_REDUCTION_HPP_
