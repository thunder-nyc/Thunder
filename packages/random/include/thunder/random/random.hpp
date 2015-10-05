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

#ifndef THUNDER_RANDOM_RANDOM_HPP_
#define THUNDER_RANDOM_RANDOM_HPP_

#include <limits>
#include <random>

namespace thunder {
namespace random {

template < typename T, typename G = ::std::mt19937,
           typename I = int, typename F = double >
class Random {
 public:
  typedef T tensor_type;
  typedef G generator_type;
  typedef I integer_type;
  typedef F float_type;
  typedef typename T::allocator_type allocator_type;
  typedef typename T::value_type value_type;
  typedef typename T::size_storage size_storage;
  typedef typename T::size_type size_type;

  template < typename... U >
  explicit Random(U... u);

  const G& generator() const;
  G& generator();

  const T& random(const T &t, I a = 0, I b = ::std::numeric_limits< I >::max());
  T& random(T &t, I a = 0, I b = ::std::numeric_limits< I >::max());

  const T& uniform(const T &t, F a = 0.0, F b = 1.0);
  T& uniform(T &t, F a = 0.0, F b = 1.0);

  const T& bernoulli(const T &t, F p = 0.5);
  T& bernoulli(T &t, F p = 0.5);

  const T& binomial(const T &t, I s = 1, F p = 0.5);
  T& binomial(T &t, I s = 1, F p = 0.5);

  const T& negativeBinomial(const T &t, I k = 1, F p = 0.5);
  T& negativeBinomial(T &t, I k = 1, F p = 0.5);

  const T& geometric(const T &t, F p = 0.5);
  T& geometric(T &t, F p = 0.5);

  const T& poisson(const T &t, F mean = 1.0);
  T& poisson(T &t, F mean = 1.0);

  const T& exponential(const T &t, F lambda = 1.0);
  T& exponential(T &t, F lambda = 1.0);

  const T& gamma(const T &t, F alpha = 1.0, F beta = 1.0);
  T& gamma(T &t, F alpha = 1.0, F beta = 1.0);

  const T& weibull(const T &t, F a = 1.0, F b = 1.0);
  T& weibull(T &t, F a = 1.0, F b = 1.0);

  const T& extremeValue(const T &t, F a = 0.0, F b = 1.0);
  T& extremeValue(T &t, F a = 0.0, F b = 1.0);

  const T& normal(const T &t, F mean = 0.0, F stddev = 1.0);
  T& normal(T &t, F mean = 0.0, F stddev = 1.0);

  const T& logNormal(const T &t, F m = 0.0, F s = 1.0);
  T& logNormal(T &t, F m = 0.0, F s = 1.0);

  const T& chiSquared(const T &t, F n = 1.0);
  T& chiSquared(T &t, F n = 1.0);

  const T& cauchy(const T &t, F a = 0.0, F b = 1.0);
  T& cauchy(T &t, F a = 0.0, F b = 1.0);

  const T& fisherF(const T &t, F m = 1.0, F n = 1.0);
  T& fisherF(T &t, F m = 1.0, F n = 1.0);

  const T& studentT(const T &t, F n = 1.0);
  T& studentT(T &t, F n = 1.0);

  const T& randperm(const T &t);
  T& randperm(T &t);
  T randperm(size_type size, allocator_type alloc = allocator_type());

 private:
  G generator_;
};

}  // namespace random
}  // namespace thunder

#endif  // THUNDER_RANDOM_RANDOM_HPP_
