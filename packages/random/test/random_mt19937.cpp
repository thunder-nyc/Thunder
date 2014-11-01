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

#include "thunder/random.hpp"

#include <random>

#include "gtest/gtest.h"

namespace thunder {
namespace {

#define TEST_RANDOM_FUNC(FUNC, DIST)                                    \
  template < typename R >                                               \
  void FUNC ## Test() {                                                 \
    typedef typename R::tensor_type T;                                  \
    typedef typename R::generator_type G;                               \
    typedef typename R::integer_type I;                                 \
    typedef typename R::float_type F;                                   \
                                                                        \
    R rand(typename G::result_type(91857));                             \
    G gen(91857);                                                       \
                                                                        \
    DIST dist1;                                                         \
    T t1(7, 33, 21);                                                    \
    rand.FUNC(t1);                                                      \
    for (typename T::reference_iterator begin = t1.reference_begin(),   \
             end = t1.reference_end(); begin != end; ++begin) {         \
      EXPECT_EQ(static_cast< typename T::value_type >(dist1(gen)), *begin); \
    }                                                                   \
                                                                        \
    DIST dist2;                                                         \
    T t2 = rand.FUNC({5, 18, 7});                                       \
    for (typename T::reference_iterator begin = t2.reference_begin(),   \
             end = t2.reference_end(); begin != end; ++begin) {         \
      EXPECT_EQ(static_cast< typename T::value_type >(dist2(gen)), *begin); \
    }                                                                   \
  }                                                                     \
                                                                        \
  TEST(RandomTest, FUNC ## VerifyTest) {                                \
    FUNC ## Test< DoubleRandom >();                                     \
    FUNC ## Test< FloatRandom >();                                      \
    FUNC ## Test< SizeRandom >();                                       \
  }

TEST_RANDOM_FUNC(random, ::std::uniform_int_distribution< I >);
TEST_RANDOM_FUNC(uniform, ::std::uniform_real_distribution< F >);
TEST_RANDOM_FUNC(bernoulli, ::std::bernoulli_distribution);
TEST_RANDOM_FUNC(binomial, ::std::binomial_distribution< I >);
TEST_RANDOM_FUNC(negativeBinomial, ::std::negative_binomial_distribution< I >);
TEST_RANDOM_FUNC(geometric, ::std::geometric_distribution< I >);
TEST_RANDOM_FUNC(poisson, ::std::poisson_distribution< I >);
TEST_RANDOM_FUNC(exponential, ::std::exponential_distribution< F >);
TEST_RANDOM_FUNC(gamma, ::std::gamma_distribution< F >);
TEST_RANDOM_FUNC(weibull, ::std::weibull_distribution< F >);
TEST_RANDOM_FUNC(extremeValue, ::std::extreme_value_distribution< F >);
TEST_RANDOM_FUNC(normal, ::std::normal_distribution< F >);
TEST_RANDOM_FUNC(logNormal, ::std::lognormal_distribution< F >);
TEST_RANDOM_FUNC(chiSquared, ::std::chi_squared_distribution< F >);
TEST_RANDOM_FUNC(cauchy, ::std::cauchy_distribution< F >);
TEST_RANDOM_FUNC(fisherF, ::std::fisher_f_distribution< F >);
TEST_RANDOM_FUNC(studentT, ::std::student_t_distribution< F >);

#undef TEST_RANDOM_FUNC

}  // namespace
}  // namespace thunder
