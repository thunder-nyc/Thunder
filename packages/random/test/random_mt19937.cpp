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
#include <unordered_set>

#include "gtest/gtest.h"
#include "thunder/exception.hpp"

namespace thunder {
namespace {

#define TEST_RANDOM_FUNC(FUNC, DIST)                                    \
  template < typename R >                                               \
  void FUNC ## Test() {                                                 \
    typedef typename R::tensor_type T;                                  \
    typedef typename R::generator_type G;                               \
                                                                        \
    R rand(91857);                                                      \
    G gen(91857);                                                       \
                                                                        \
    DIST dist1;                                                         \
    T t1(7, 11, 21);                                                    \
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
                                                                        \
    DIST dist3;                                                         \
    T t3 = rand.FUNC(T({5, 18, 7}, {271, 15, 2}));                      \
    for (typename T::reference_iterator begin = t3.reference_begin(),   \
             end = t3.reference_end(); begin != end; ++begin) {         \
      EXPECT_EQ(static_cast< typename T::value_type >(dist3(gen)), *begin); \
    }                                                                   \
  }                                                                     \
                                                                        \
  TEST(RandomTest, FUNC ## Test) {                                      \
    FUNC ## Test< DoubleRandom >();                                     \
    FUNC ## Test< FloatRandom >();                                      \
    FUNC ## Test< SizeRandom >();                                       \
  }

TEST_RANDOM_FUNC(
    random, ::std::uniform_int_distribution< typename R::integer_type >);
TEST_RANDOM_FUNC(
    uniform, ::std::uniform_real_distribution< typename R::float_type >);
TEST_RANDOM_FUNC(
    bernoulli, ::std::bernoulli_distribution);
TEST_RANDOM_FUNC(
    binomial, ::std::binomial_distribution< typename R::integer_type >);
TEST_RANDOM_FUNC(
    negativeBinomial,
    ::std::negative_binomial_distribution< typename R::integer_type >);
TEST_RANDOM_FUNC(
    geometric, ::std::geometric_distribution< typename R::integer_type >);
TEST_RANDOM_FUNC(
    poisson, ::std::poisson_distribution< typename R::integer_type >);
TEST_RANDOM_FUNC(
    exponential, ::std::exponential_distribution< typename R::float_type >);
TEST_RANDOM_FUNC(
    gamma, ::std::gamma_distribution< typename R::float_type >);
TEST_RANDOM_FUNC(
    weibull, ::std::weibull_distribution< typename R::float_type >);
TEST_RANDOM_FUNC(
    extremeValue, ::std::extreme_value_distribution< typename R::float_type >);
TEST_RANDOM_FUNC(
    normal, ::std::normal_distribution< typename R::float_type >);
TEST_RANDOM_FUNC(
    logNormal, ::std::lognormal_distribution< typename R::float_type >);
TEST_RANDOM_FUNC(
    chiSquared, ::std::chi_squared_distribution< typename R::float_type >);
TEST_RANDOM_FUNC(
    cauchy, ::std::cauchy_distribution< typename R::float_type >);
TEST_RANDOM_FUNC(
    fisherF, ::std::fisher_f_distribution< typename R::float_type >);
TEST_RANDOM_FUNC(
    studentT, ::std::student_t_distribution< typename R::float_type >);

#undef TEST_RANDOM_FUNC

template < typename R >
void randpermTest() {
  typedef typename R::tensor_type T;
  typedef typename R::generator_type G;

  R rand(91857);
  G gen(91857);
  ::std::unordered_set< typename T::value_type > set;

  T t1(33);
  rand.randperm(t1);
  set.clear();
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    set.insert(*begin);
  }
  for (typename T::size_type i = 0; i < 33; ++i) {
    EXPECT_NE(set.find(static_cast< typename T::value_type >(i)), set.end());
  }

  T t2 = rand.randperm(65);
  set.clear();
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    set.insert(*begin);
  }
  for (typename T::size_type i = 0; i < 65; ++i) {
    EXPECT_NE(set.find(static_cast< typename T::value_type >(i)), set.end());
  }

  T t3(2, 7, 8);
  EXPECT_THROW(rand.randperm(t3), invalid_argument);
}

TEST(RandomTest, randpermTest) {
  randpermTest< DoubleRandom >();
  randpermTest< FloatRandom >();
  randpermTest< SizeRandom >();
}

}  // namespace
}  // namespace thunder
