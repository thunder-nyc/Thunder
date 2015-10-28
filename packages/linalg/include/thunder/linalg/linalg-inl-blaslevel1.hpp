/*
 * \copyright Copyright 2014-2015 Xiang Zhang All Rights Reserved.
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

#ifndef THUNDER_LINALG_LINALG_INL_BLASLEVEL1_HPP_
#define THUNDER_LINALG_LINALG_INL_BLASLEVEL1_HPP_

#include "thunder/linalg/linalg.hpp"
#include "thunder/linalg/linalg-inl.hpp"

#include "thunder/linalg/math.hpp"

namespace thunder {
namespace linalg {

template < typename T, typename H >
const T& Linalg< T, H >::asum(const T &x, const T &r) {
  return math::asum(this, x, r);
}

template < typename T, typename H >
const T& Linalg< T, H >::axpy(const T &x, const T &y, const value_type &a) {
  return math::axpy(this, x, y, a);
}

template < typename T, typename H >
const T& Linalg< T, H >::copy(const T &x, const T &r) {
  return math::copy(this, x, r);
}

template < typename T, typename H >
const T& Linalg< T, H >::dot(const T &x, const T &y, const T &r) {
  return math::dot(this, x, y, r);
}

template < typename T, typename H >
const T& Linalg< T, H >::dotc(const T &x, const T &y, const T &r) {
  return math::dotc(this, x, y, r);
}

template < typename T, typename H >
const T& Linalg< T, H >::nrm2(const T &x, const T &r) {
  return math::nrm2(this, x, r);
}

template < typename T, typename H >
const T& Linalg< T, H >::rot(
    const T &x, const T &y, const value_type &c, const value_type &s) {
  return math::rot(this, x, y, c, s);
}

template < typename T, typename H >
const T& Linalg< T, H >::rotm(const T &x, const T &y, const T &p) {
  return math::rotm(this, x, y, p);
}

template < typename T, typename H >
const T& Linalg< T, H >::scal(const T &x, const value_type &a) {
  return math::scal(this, x, a);
}

template < typename T, typename H >
const T& Linalg< T, H >::swap(const T &x, const T &y) {
  return math::swap(this, x, y);
}

template < typename T, typename H >
const typename Linalg< T, H >::size_tensor& Linalg< T, H >::iamax(
    const T &x, const size_tensor &r) {
  return math::iamax(this, x, r);
}

}  //  namespace linalg
}  //  namespace thunder

#endif  // THUNDER_LINALG_LINALG_INL_BLASLEVEL1_HPP_
