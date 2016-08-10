/*
 * \copyright Copyright 2014-2016 Xiang Zhang All Rights Reserved.
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

#include "thunder/linalg.hpp"

#include <complex>
#include <memory>
#include <random>

#include "gtest/gtest.h"
#include "thunder/exception.hpp"
#include "thunder/linalg/cxxblas.hpp"
#include "thunder/random.hpp"
#include "thunder/tensor.hpp"

namespace thunder {
namespace linalg {
namespace {

template < typename T >
const T& uniformDist(const T& t) {
  Random< typename T::real_tensor > r;
  r.uniform(t.viewReal());
  try {
    r.uniform(t.viewImag());
  } catch (const domain_error &e) {
    // Do nothing for an real tensor type
  }
  return t;
}
template < typename T >
T& uniformDist(T& t) {
  return const_cast< T& >(uniformDist(const_cast< const T& >(t)));
}

template < typename L >
void gbmvTest() {
  typedef typename L::tensor_type T;
  typedef typename T::value_type D;

  L linalg;
  D alpha = D(1.57);
  D beta = D(1.36);
  typename T::size_type kl = 1;
  typename T::size_type ku = 2;

  T a1 = uniformDist(T(7, kl + ku + 1));
  T x1 = uniformDist(T(10));
  T y1 = linalg.gbmv(a1, x1, alpha, beta, kl, ku);
  EXPECT_EQ(a1.dimension() - 1, y1.dimension());
  EXPECT_EQ(a1.size(0), y1.size(0));
  T v1 = T(y1.size()).zero();
  cxxblas::gbmv(y1.size(0), x1.size(0), a1.data(), x1.data(), v1.data(), alpha,
                beta, kl, ku, a1.stride(0), x1.stride(0), v1.stride(0));
  for (typename T::reference_iterator begin = v1.reference_begin(),
           end = v1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(y1(begin.position()), *begin);
  }

  T a2 = uniformDist(T(3, 9, 7, kl + ku + 1));
  T x2 = uniformDist(T(3, 9, 10));
  T y2 = linalg.gbmv(a2, x2, alpha, beta, kl, ku);
  EXPECT_EQ(a2.dimension() - 1, y2.dimension());
  for (typename T::dim_type i = 0; i < y2.dimension(); ++i) {
    EXPECT_EQ(a2.size(i), y2.size(i));
  }
  T v2 = T(y2.size()).zero();
  T x2_select = x2.select(x2.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x2_select.reference_begin(),
           end = x2_select.reference_end(); begin != end; ++begin) {
    cxxblas::gbmv(
        y2.size(y2.dimension() - 1), x2.size(x2.dimension() - 1),
        a2[begin.position()].data(), x2[begin.position()].data(),
        v2[begin.position()].data(), alpha, beta, kl, ku,
        a2.stride(a2.dimension() - 2), x2.stride(x2.dimension() - 1),
        v2.stride(v2.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v2.reference_begin(),
           end = v2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(y2(begin.position()), *begin);
  }

  T a3 = uniformDist(T({3, 9, 7, kl + ku + 1}, {1335, 148, 21, 1}));
  T x3 = uniformDist(T({3, 9, 10}, {193, 21, 2}));
  T y3 = linalg.gbmv(a3, x3, alpha, beta, kl, ku);
  EXPECT_EQ(a3.dimension() - 1, y3.dimension());
  for (typename T::dim_type i = 0; i < y3.dimension(); ++i) {
    EXPECT_EQ(a3.size(i), y3.size(i));
  }
  T v3 = T(y3.size()).zero();
  T x3_select = x3.select(x3.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x3_select.reference_begin(),
           end = x3_select.reference_end(); begin != end; ++begin) {
    cxxblas::gbmv(
        y3.size(y3.dimension() - 1), x3.size(x3.dimension() - 1),
        a3[begin.position()].data(), x3[begin.position()].data(),
        v3[begin.position()].data(), alpha, beta, kl, ku,
        a3.stride(a3.dimension() - 2), x3.stride(x3.dimension() - 1),
        v3.stride(v3.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v3.reference_begin(),
           end = v3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(y3(begin.position()), *begin);
  }
}

TEST(LinalgTest, gbmvTest) {
  gbmvTest< DoubleLinalg >();
  gbmvTest< FloatLinalg >();
  gbmvTest< DoubleComplexLinalg >();
  gbmvTest< FloatComplexLinalg >();
}

template < typename L >
void gemvTest() {
  typedef typename L::tensor_type T;
  typedef typename T::value_type D;

  L linalg;
  D alpha = D(1.57);
  D beta = D(1.36);

  T a1 = uniformDist(T(7, 10));
  T x1 = uniformDist(T(10));
  T y1 = linalg.gemv(a1, x1, alpha, beta);
  EXPECT_EQ(a1.dimension() - 1, y1.dimension());
  EXPECT_EQ(a1.size(0), y1.size(0));
  T v1 = T(y1.size()).zero();
  cxxblas::gemv(a1.size(0), a1.size(1), a1.data(), x1.data(), v1.data(), alpha,
                beta, a1.stride(0), x1.stride(0), v1.stride(0));
  for (typename T::reference_iterator begin = v1.reference_begin(),
           end = v1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(y1(begin.position()), *begin);
  }

  T a2 = uniformDist(T(3, 9, 7, 10));
  T x2 = uniformDist(T(3, 9, 10));
  T y2 = linalg.gemv(a2, x2, alpha, beta);
  EXPECT_EQ(a2.dimension() - 1, y2.dimension());
  for (typename T::dim_type i = 0; i < y2.dimension(); ++i) {
    EXPECT_EQ(a2.size(i), y2.size(i));
  }
  T v2 = T(y2.size()).zero();
  T x2_select = x2.select(x2.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x2_select.reference_begin(),
           end = x2_select.reference_end(); begin != end; ++begin) {
    cxxblas::gemv(
        a2.size(a2.dimension() - 2), a2.size(a2.dimension() - 1),
        a2[begin.position()].data(), x2[begin.position()].data(),
        v2[begin.position()].data(), alpha, beta, a2.stride(a2.dimension() - 2),
        x2.stride(x2.dimension() - 1), v2.stride(v2.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v2.reference_begin(),
           end = v2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(y2(begin.position()), *begin);
  }

  T a3 = uniformDist(T({3, 9, 7, 10}, {1335, 148, 21, 1}));
  T x3 = uniformDist(T({3, 9, 10}, {193, 21, 2}));
  T y3 = linalg.gemv(a3, x3, alpha, beta);
  EXPECT_EQ(a3.dimension() - 1, y3.dimension());
  for (typename T::dim_type i = 0; i < y3.dimension(); ++i) {
    EXPECT_EQ(a3.size(i), y3.size(i));
  }
  T v3 = T(y3.size()).zero();
  T x3_select = x3.select(x3.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x3_select.reference_begin(),
           end = x3_select.reference_end(); begin != end; ++begin) {
    cxxblas::gemv(
        a3.size(a3.dimension() - 2), a3.size(a3.dimension() - 1),
        a3[begin.position()].data(), x3[begin.position()].data(),
        v3[begin.position()].data(), alpha, beta, a3.stride(a3.dimension() - 2),
        x3.stride(x3.dimension() - 1), v3.stride(v3.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v3.reference_begin(),
           end = v3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(y3(begin.position()), *begin);
  }
}

TEST(LinalgTest, gemvTest) {
  gemvTest< DoubleLinalg >();
  gemvTest< FloatLinalg >();
  gemvTest< DoubleComplexLinalg >();
  gemvTest< FloatComplexLinalg >();
}

template < typename L >
void gerTest() {
  typedef typename L::tensor_type T;
  typedef typename T::value_type D;

  L linalg;
  D alpha = D(1.57);

  T x1 = uniformDist(T(7));
  T y1 = uniformDist(T(10));
  T a1 = linalg.ger(x1, y1, alpha);
  EXPECT_EQ(x1.dimension() + 1, a1.dimension());
  T v1 = T(a1.size()).zero();
  cxxblas::ger(x1.size(0), y1.size(0), x1.data(), y1.data(), v1.data(), alpha,
               x1.stride(0), y1.stride(0), v1.stride(0));
  for (typename T::reference_iterator begin = v1.reference_begin(),
           end = v1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a1(begin.position()), *begin);
  }

  T x2 = uniformDist(T(3, 9, 7));
  T y2 = uniformDist(T(3, 9, 10));
  T a2 = linalg.ger(x2, y2, alpha);
  EXPECT_EQ(x2.dimension() + 1, a2.dimension());
  for (typename T::dim_type i = 0; i < x2.dimension(); ++i) {
    EXPECT_EQ(x2.size(i), a2.size(i));
  }
  EXPECT_EQ(y2.size(y2.dimension() - 1), a2.size(a2.dimension() - 1));
  T v2 = T(a2.size()).zero();
  T x2_select = x2.select(x2.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x2_select.reference_begin(),
           end = x2_select.reference_end(); begin != end; ++begin) {
    cxxblas::ger(
        x2.size(x2.dimension() - 1), y2.size(y2.dimension() - 1),
        x2[begin.position()].data(), y2[begin.position()].data(),
        v2[begin.position()].data(), alpha, x2.stride(x2.dimension() - 1),
        y2.stride(y2.dimension() - 1), v2.stride(v2.dimension() - 2));
  }
  for (typename T::reference_iterator begin = v2.reference_begin(),
           end = v2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a2(begin.position()), *begin);
  }

  T x3 = uniformDist(T({3, 9, 7}, {47, 15, 2}));
  T y3 = uniformDist(T({3, 9, 10}, {69, 22, 2}));
  T a3 = linalg.ger(x3, y3, alpha);
  EXPECT_EQ(x3.dimension() + 1, a3.dimension());
  for (typename T::dim_type i = 0; i < x3.dimension(); ++i) {
    EXPECT_EQ(x3.size(i), a3.size(i));
  }
  EXPECT_EQ(y3.size(y3.dimension() - 1), a3.size(a3.dimension() - 1));
  T v3 = T(a3.size()).zero();
  T x3_select = x3.select(x3.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x3_select.reference_begin(),
           end = x3_select.reference_end(); begin != end; ++begin) {
    cxxblas::ger(
        x3.size(x3.dimension() - 1), y3.size(y3.dimension() - 1),
        x3[begin.position()].data(), y3[begin.position()].data(),
        v3[begin.position()].data(), alpha, x3.stride(x3.dimension() - 1),
        y3.stride(y3.dimension() - 1), v3.stride(v3.dimension() - 2));
  }
  for (typename T::reference_iterator begin = v3.reference_begin(),
           end = v3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a3(begin.position()), *begin);
  }
}

TEST(LinalgTest, gerTest) {
  gerTest< DoubleLinalg >();
  gerTest< FloatLinalg >();
  gerTest< DoubleComplexLinalg >();
  gerTest< FloatComplexLinalg >();
}

template < typename L >
void gercTest() {
  typedef typename L::tensor_type T;
  typedef typename T::value_type D;

  L linalg;
  D alpha = D(1.57);

  T x1 = uniformDist(T(7));
  T y1 = uniformDist(T(10));
  T a1 = linalg.gerc(x1, y1, alpha);
  EXPECT_EQ(x1.dimension() + 1, a1.dimension());
  T v1 = T(a1.size()).zero();
  cxxblas::gerc(x1.size(0), y1.size(0), x1.data(), y1.data(), v1.data(), alpha,
               x1.stride(0), y1.stride(0), v1.stride(0));
  for (typename T::reference_iterator begin = v1.reference_begin(),
           end = v1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a1(begin.position()), *begin);
  }

  T x2 = uniformDist(T(3, 9, 7));
  T y2 = uniformDist(T(3, 9, 10));
  T a2 = linalg.gerc(x2, y2, alpha);
  EXPECT_EQ(x2.dimension() + 1, a2.dimension());
  for (typename T::dim_type i = 0; i < x2.dimension(); ++i) {
    EXPECT_EQ(x2.size(i), a2.size(i));
  }
  EXPECT_EQ(y2.size(y2.dimension() - 1), a2.size(a2.dimension() - 1));
  T v2 = T(a2.size()).zero();
  T x2_select = x2.select(x2.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x2_select.reference_begin(),
           end = x2_select.reference_end(); begin != end; ++begin) {
    cxxblas::gerc(
        x2.size(x2.dimension() - 1), y2.size(y2.dimension() - 1),
        x2[begin.position()].data(), y2[begin.position()].data(),
        v2[begin.position()].data(), alpha, x2.stride(x2.dimension() - 1),
        y2.stride(y2.dimension() - 1), v2.stride(v2.dimension() - 2));
  }
  for (typename T::reference_iterator begin = v2.reference_begin(),
           end = v2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a2(begin.position()), *begin);
  }

  T x3 = uniformDist(T({3, 9, 7}, {47, 15, 2}));
  T y3 = uniformDist(T({3, 9, 10}, {69, 22, 2}));
  T a3 = linalg.gerc(x3, y3, alpha);
  EXPECT_EQ(x3.dimension() + 1, a3.dimension());
  for (typename T::dim_type i = 0; i < x3.dimension(); ++i) {
    EXPECT_EQ(x3.size(i), a3.size(i));
  }
  EXPECT_EQ(y3.size(y3.dimension() - 1), a3.size(a3.dimension() - 1));
  T v3 = T(a3.size()).zero();
  T x3_select = x3.select(x3.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x3_select.reference_begin(),
           end = x3_select.reference_end(); begin != end; ++begin) {
    cxxblas::gerc(
        x3.size(x3.dimension() - 1), y3.size(y3.dimension() - 1),
        x3[begin.position()].data(), y3[begin.position()].data(),
        v3[begin.position()].data(), alpha, x3.stride(x3.dimension() - 1),
        y3.stride(y3.dimension() - 1), v3.stride(v3.dimension() - 2));
  }
  for (typename T::reference_iterator begin = v3.reference_begin(),
           end = v3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a3(begin.position()), *begin);
  }
}

TEST(LinalgTest, gercTest) {
  gercTest< DoubleLinalg >();
  gercTest< FloatLinalg >();
  gercTest< DoubleComplexLinalg >();
  gercTest< FloatComplexLinalg >();
}

}  // namespace
}  // namespace linalg
}  // namespace thunder
