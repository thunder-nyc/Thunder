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

#ifndef THUNDER_RANDOM_RANDOM_INL_HPP_
#define THUNDER_RANDOM_RANDOM_INL_HPP_

#include "thunder/random/random.hpp"

#include "thunder/random/math.hpp"

namespace thunder {
namespace random {

template < typename T, typename G, typename I, typename F >
template < typename... U >
Random< T, G, I, F >::Random(U... u) : generator_(u...) {}

template < typename T, typename G, typename I, typename F >
const G& Random< T, G, I, F >::generator() const {
  return generator_;
}

template < typename T, typename G, typename I, typename F >
G& Random< T, G, I, F >::generator() {
  return const_cast< G& >(const_cast< const Random* >(this)->generator());
}

template < typename T, typename G, typename I, typename F >
const T& Random< T, G, I, F >::random(const T &t, I a, I b) {
  return math::random< Random >(this, t, a, b);
}

template < typename T, typename G, typename I, typename F >
T& Random< T, G, I, F >::random(T &t, I a, I b) {
  return const_cast< T& >(random(const_cast< const T& >(t), a, b));
}

template < typename T, typename G, typename I, typename F >
const T& Random< T, G, I, F >::uniform(const T &t, F a, F b) {
  return math::uniform< Random >(this, t, a, b);
}

template < typename T, typename G, typename I, typename F >
T& Random< T, G, I, F >::uniform(T &t, F a, F b) {
  return const_cast< T& >(uniform(const_cast< const T& >(t), a, b));
}

template < typename T, typename G, typename I, typename F >
const T& Random< T, G, I, F >::bernoulli(const T &t, F p) {
  return math::bernoulli< Random >(this, t, p);
}

template < typename T, typename G, typename I, typename F >
T& Random< T, G, I, F >::bernoulli(T &t, F p) {
  return const_cast< T& >(bernoulli(const_cast< const T& >(t), p));
}

template < typename T, typename G, typename I, typename F >
const T& Random< T, G, I, F >::binomial(const T &t, I s, F p) {
  return math::binomial< Random >(this, t, s, p);
}

template < typename T, typename G, typename I, typename F >
T& Random< T, G, I, F >::binomial(T &t, I s, F p) {
  return const_cast< T& >(binomial(const_cast< const T& >(t), s, p));
}

template< typename T, typename G, typename I, typename F >
const T& Random< T, G, I, F >::negativeBinomial(const T &t, I k, F p) {
  return math::negativeBinomial< Random >(this, t, k, p);
}

template< typename T, typename G, typename I, typename F >
T& Random< T, G, I, F >::negativeBinomial(T &t, I k, F p) {
  return const_cast< T& >(negativeBinomial(const_cast< const T& >(t), k, p));
                                                           }

template< typename T, typename G, typename I, typename F >
const T& Random< T, G, I, F >::geometric(const T &t, F p) {
  return math::geometric< Random >(this, t, p);
}

template< typename T, typename G, typename I, typename F >
T& Random< T, G, I, F >::geometric(T &t, F p) {
  return const_cast< T& >(geometric(const_cast< const T& >(t), p));
}

template< typename T, typename G, typename I, typename F >
const T& Random< T, G, I, F >::poisson(const T &t, F mean) {
  return math::poisson< Random >(this, t, mean);
}

template< typename T, typename G, typename I, typename F >
T& Random< T, G, I, F >::poisson(T &t, F mean) {
  return const_cast< T& >(poisson(const_cast< const T& >(t), mean));
                                                }

template< typename T, typename G, typename I, typename F >
const T& Random< T, G, I, F >::exponential(const T &t, F lambda) {
  return math::exponential< Random >(this, t, lambda);
}

template< typename T, typename G, typename I, typename F >
T& Random< T, G, I, F >::exponential(T &t, F lambda) {
  return const_cast< T& >(exponential(const_cast< const T& >(t), lambda));
}

template< typename T, typename G, typename I, typename F >
const T& Random< T, G, I, F >::gamma(const T &t, F alpha, F beta) {
  return math::gamma< Random >(this, t, alpha, beta);
}

template< typename T, typename G, typename I, typename F >
T& Random< T, G, I, F >::gamma(T &t, F alpha, F beta) {
  return const_cast< T& >(gamma(const_cast< const T& >(t), alpha, beta));
}

template< typename T, typename G, typename I, typename F >
const T& Random< T, G, I, F >::weibull(const T &t, F a, F b) {
  return math::weibull< Random >(this, t, a, b);
}

template< typename T, typename G, typename I, typename F >
T& Random< T, G, I, F >::weibull(T &t, F a, F b) {
  return const_cast< T& >(weibull(const_cast< const T& >(t), a, b));
}

template< typename T, typename G, typename I, typename F >
const T& Random< T, G, I, F >::extremeValue(const T &t, F a, F b) {
  return math::extremeValue< Random >(this, t, a, b);
}

template< typename T, typename G, typename I, typename F >
T& Random< T, G, I, F >::extremeValue(T &t, F a, F b) {
  return const_cast< T& >(extremeValue(const_cast< const T& >(t), a, b));
}

template< typename T, typename G, typename I, typename F >
const T& Random< T, G, I, F >::normal(const T &t, F mean, F stddev) {
  return math::normal< Random >(this, t, mean, stddev);
}

template< typename T, typename G, typename I, typename F >
T& Random< T, G, I, F >::normal(T &t, F mean, F stddev) {
  return const_cast< T& >(normal(const_cast< const T& >(t), mean, stddev));
}

template< typename T, typename G, typename I, typename F >
const T& Random< T, G, I, F >::logNormal(const T &t, F m, F s) {
  return math::logNormal< Random >(this, t, m, s);
}

template< typename T, typename G, typename I, typename F >
T& Random< T, G, I, F >::logNormal(T &t, F m, F s) {
  return const_cast< T& >(logNormal(const_cast< const T& >(t), m, s));
}

template< typename T, typename G, typename I, typename F >
const T& Random< T, G, I, F >::chiSquared(const T &t, F n) {
  return math::chiSquared< Random >(this, t, n);
}

template< typename T, typename G, typename I, typename F >
T& Random< T, G, I, F >::chiSquared(T &t, F n) {
  return const_cast< T& >(chiSquared(const_cast< const T& >(t), n));
}

template< typename T, typename G, typename I, typename F >
const T& Random< T, G, I, F >::cauchy(const T &t, F a, F b) {
  return math::cauchy< Random >(this, t, a, b);
}

template< typename T, typename G, typename I, typename F >
T& Random< T, G, I, F >::cauchy(T &t, F a, F b) {
  return const_cast< T& >(cauchy(const_cast< const T& >(t), a, b));
}

template< typename T, typename G, typename I, typename F >
const T& Random< T, G, I, F >::fisherF(const T &t, F m, F n) {
  return math::fisherF< Random >(this, t, m, n);
}

template< typename T, typename G, typename I, typename F >
T& Random< T, G, I, F >::fisherF(T &t, F m, F n) {
  return const_cast< T& >(fisherF(const_cast< const T& >(t), m, n));
}

template< typename T, typename G, typename I, typename F >
const T& Random< T, G, I, F >::studentT(const T &t, F n) {
  return math::studentT< Random >(this, t, n);
}

template< typename T, typename G, typename I, typename F >
T& Random< T, G, I, F >::studentT(T &t, F n) {
  return const_cast< T& >(studentT(const_cast< const T& >(t), n));
}

template< typename T, typename G, typename I, typename F >
const T& Random< T, G, I, F >::randperm(const T &t) {
  return math::randperm< Random >(this, t);
}

template< typename T, typename G, typename I, typename F >
T& Random< T, G, I, F >::randperm(T &t) {
  return const_cast< T& >(randperm(const_cast< const T& >(t)));
}

template< typename T, typename G, typename I, typename F >
T Random< T, G, I, F >::randperm(size_type size) {
  T t(size);
  return randperm(t);
}

}  // namespace random
}  // namespace thunder

#endif  // THUNDER_RANDOM_RANDOM_INL_HPP_
