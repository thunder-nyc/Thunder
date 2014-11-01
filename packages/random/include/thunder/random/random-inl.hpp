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

#include <limits>
#include <random>

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
void Random< T, G, I, F >::random(const T &t, I a, I b);

template < typename T, typename G, typename I, typename F >
T Random< T, G, I, F >::random(const size_storage &size, I a, I b);

template < typename T, typename G, typename I, typename F >
void Random< T, G, I, F >::uniform(const T &t, F a, F b);

template < typename T, typename G, typename I, typename F >
T Random< T, G, I, F >::uniform(const size_storage &size, F a, F b);

template < typename T, typename G, typename I, typename F >
void Random< T, G, I, F >::bernoulli(const T &t, F p);

template < typename T, typename G, typename I, typename F >
T Random< T, G, I, F >::bernoulli(const size_storage &size, F p);

template < typename T, typename G, typename I, typename F >
void Random< T, G, I, F >::binomial(const T &t, I s, F p);

template < typename T, typename G, typename I, typename F >
T Random< T, G, I, F >::binomial(const size_storage &size, I s, F p);

template< typename T, typename G, typename I, typename F >
void Random< T, G, I, F >::negativeBinomial(const T &t, I k, F p);

template< typename T, typename G, typename I, typename F >
T Random< T, G, I, F >::negativeBinomial(const size_storage &size, I k, F p);

template< typename T, typename G, typename I, typename F >
void Random< T, G, I, F >::geometric(const T &t, F p);

template< typename T, typename G, typename I, typename F >
T Random< T, G, I, F >::geometric(const size_storage &size, F p);

template< typename T, typename G, typename I, typename F >
void Random< T, G, I, F >::poisson(const T &t, F mean);

template< typename T, typename G, typename I, typename F >
T Random< T, G, I, F >::poisson(const size_storage &size, F mean);

template< typename T, typename G, typename I, typename F >
void Random< T, G, I, F >::exponential(const T &t, F mean);

template< typename T, typename G, typename I, typename F >
T Random< T, G, I, F >::exponential(const size_storage &size, F mean);

template< typename T, typename G, typename I, typename F >
void Random< T, G, I, F >::gamma(const T &t, F alpha, F beta);

template< typename T, typename G, typename I, typename F >
T Random< T, G, I, F >::gamma(const size_storage &size, F alpha, F beta);

template< typename T, typename G, typename I, typename F >
void Random< T, G, I, F >::weibull(const T &t, F a, F b);

template< typename T, typename G, typename I, typename F >
T Random< T, G, I, F >::weibull(const size_storage &size, F a, F b);

template< typename T, typename G, typename I, typename F >
void Random< T, G, I, F >::extremeValue(const T &t, F a, F b);

template< typename T, typename G, typename I, typename F >
T Random< T, G, I, F >::extremeValue(const size_storage &size, F a, F b);

template< typename T, typename G, typename I, typename F >
void Random< T, G, I, F >::normal(const T &t, F mean, F stddev);

template< typename T, typename G, typename I, typename F >
T Random< T, G, I, F >::normal(const size_storage &size, F mean, F stddev);

template< typename T, typename G, typename I, typename F >
void Random< T, G, I, F >::logNormal(const T &t, F m, F s);

template< typename T, typename G, typename I, typename F >
T Random< T, G, I, F >::logNormal(const size_storage &size, F m, F s);

template< typename T, typename G, typename I, typename F >
void Random< T, G, I, F >::chiSquared(const T &t, F n);

template< typename T, typename G, typename I, typename F >
T Random< T, G, I, F >::chiSquared(const size_storage &size, F n);

template< typename T, typename G, typename I, typename F >
void Random< T, G, I, F >::cauchy(const T &t, F a, F b);

template< typename T, typename G, typename I, typename F >
T Random< T, G, I, F >::cauchy(const size_storage &size, F a, F b);

template< typename T, typename G, typename I, typename F >
void Random< T, G, I, F >::fisherF(const T &t, F m, F n);

template< typename T, typename G, typename I, typename F >
T Random< T, G, I, F >::fisherF(const size_storage &size, F m, F n);

template< typename T, typename G, typename I, typename F >
void Random< T, G, I, F >::studentT(const T &t, F n);

template< typename T, typename G, typename I, typename F >
T Random< T, G, I, F >::studentT(const size_storage &size, F n);

}  // namespace random
}  // namespace thunder

#endif  // THUNDER_RANDOM_RANDOM_INL_HPP_
