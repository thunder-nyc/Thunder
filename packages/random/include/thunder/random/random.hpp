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
  typedef typename T::value_type value_type;
  typedef typename T::size_storage size_storage;

  template < typename... U >
  explicit Random(U... u);

  const G& generator() const;
  G& generator();

  void random(const T &t, I a = 0, I b = ::std::numeric_limits< I >::max);
  T random(const size_storage &size, I a = 0,
           I b = ::std::numeric_limits< I >::max);

  void uniform(const T &t, F a = 0.0, F b = 1.0);
  T uniform(const size_storage &size, F a = 0.0, F b = 1.0);

  void bernoulli(const T &t, F p = 0.5);
  T bernoulli(const size_storage &size, F p = 0.5);

  void binomial(const T &t, I s = 1, F p = 0.5);
  T binomial(const size_storage &size, I s = 1, F p = 0.5);

  void negativeBinomial(const T &t, I k = 1, F p = 0.5);
  T negativeBinomial(const size_storage &size, I k = 1, F p = 0.5);

  void geometric(const T &t, F p = 0.5);
  T geometric(const size_storage &size, F p = 0.5);

  void poisson(const T &t, F mean = 1.0);
  T poisson(const size_storage &size, F mean = 1.0);

  void exponential(const T &t, F lambda = 1.0);
  T exponential(const size_storage &size, F lambda = 1.0);

  void gamma(const T &t, F alpha = 1.0, F beta = 1.0);
  T gamma(const size_storage &size, F alpha = 1.0, F beta = 1.0);

  void weibull(const T &t, F a = 1.0, F b = 1.0);
  T weibull(const size_storage &size, F a = 1.0, F b = 1.0);

  void extremeValue(const T &t, F a = 0.0, F b = 1.0);
  T extremeValue(const size_storage &size, F a = 0.0, F b = 1.0);

  void normal(const T &t, F mean = 0.0, F stddev = 1.0);
  T normal(const size_storage &size, F mean = 0.0, F stddev = 1.0);

  void logNormal(const T &t, F m = 0.0, F s = 1.0);
  T logNormal(const size_storage &size, F m = 0.0, F s = 1.0);

  void chiSquared(const T &t, F n = 1.0);
  T chiSquared(const size_storage &size, F n = 1.0);

  void cauchy(const T &t, F a = 0.0, F b = 1.0);
  T cauchy(const size_storage &size, F a = 0.0, F b = 1.0);

  void fisherF(const T &t, F m = 1.0, F n = 1.0);
  T fisherF(const size_storage &size, F m = 1.0, F n = 1.0);

  void studentT(const T &t, F n = 1.0);
  T studentT(const size_storage &size, F n = 1.0);

 private:
  G generator_;
};

}  // namespace random
}  // namespace thunder

#endif  // THUNDER_RANDOM_RANDOM_HPP_
