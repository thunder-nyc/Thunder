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
    // Do nothing for a real tensor type
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

template < typename L >
void hbmvTest() {
  typedef typename L::tensor_type T;
  typedef typename T::value_type D;

  L linalg;
  D alpha = D(1.57);
  D beta = D(1.36);
  typename T::size_type k = 2;

  T a1 = uniformDist(T(7, k + 1));
  T x1 = uniformDist(T(7));
  T y1 = linalg.hbmv(a1, x1, alpha, beta, k);
  EXPECT_EQ(a1.dimension() - 1, y1.dimension());
  EXPECT_EQ(a1.size(0), y1.size(0));
  T v1 = T(y1.size()).zero();
  cxxblas::hbmv(x1.size(0), a1.data(), x1.data(), v1.data(), alpha,
                beta, k, a1.stride(0), x1.stride(0), v1.stride(0));
  for (typename T::reference_iterator begin = v1.reference_begin(),
           end = v1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(y1(begin.position()), *begin);
  }

  T a2 = uniformDist(T(3, 9, 7, k + 1));
  T x2 = uniformDist(T(3, 9, 7));
  T y2 = linalg.hbmv(a2, x2, alpha, beta, k);
  EXPECT_EQ(a2.dimension() - 1, y2.dimension());
  for (typename T::dim_type i = 0; i < y2.dimension(); ++i) {
    EXPECT_EQ(a2.size(i), y2.size(i));
  }
  T v2 = T(y2.size()).zero();
  T x2_select = x2.select(x2.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x2_select.reference_begin(),
           end = x2_select.reference_end(); begin != end; ++begin) {
    cxxblas::hbmv(
        x2.size(x2.dimension() - 1), a2[begin.position()].data(),
        x2[begin.position()].data(), v2[begin.position()].data(), alpha, beta,
        k, a2.stride(a2.dimension() - 2), x2.stride(x2.dimension() - 1),
        v2.stride(v2.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v2.reference_begin(),
           end = v2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(y2(begin.position()), *begin);
  }

  T a3 = uniformDist(T({3, 9, 7, k + 1}, {1335, 148, 21, 1}));
  T x3 = uniformDist(T({3, 9, 7}, {193, 21, 2}));
  T y3 = linalg.hbmv(a3, x3, alpha, beta, k);
  EXPECT_EQ(a3.dimension() - 1, y3.dimension());
  for (typename T::dim_type i = 0; i < y3.dimension(); ++i) {
    EXPECT_EQ(a3.size(i), y3.size(i));
  }
  T v3 = T(y3.size()).zero();
  T x3_select = x3.select(x3.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x3_select.reference_begin(),
           end = x3_select.reference_end(); begin != end; ++begin) {
    cxxblas::hbmv(
        x3.size(x3.dimension() - 1), a3[begin.position()].data(),
        x3[begin.position()].data(), v3[begin.position()].data(), alpha, beta,
        k, a3.stride(a3.dimension() - 2), x3.stride(x3.dimension() - 1),
        v3.stride(v3.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v3.reference_begin(),
           end = v3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(y3(begin.position()), *begin);
  }
}

TEST(LinalgTest, hbmvTest) {
  hbmvTest< DoubleLinalg >();
  hbmvTest< FloatLinalg >();
  hbmvTest< DoubleComplexLinalg >();
  hbmvTest< FloatComplexLinalg >();
}

template < typename L >
void hemvTest() {
  typedef typename L::tensor_type T;
  typedef typename T::value_type D;

  L linalg;
  D alpha = D(1.57);
  D beta = D(1.36);

  T a1 = uniformDist(T(7, 7));
  T x1 = uniformDist(T(7));
  T y1 = linalg.hemv(a1, x1, alpha, beta);
  EXPECT_EQ(a1.dimension() - 1, y1.dimension());
  EXPECT_EQ(a1.size(0), y1.size(0));
  T v1 = T(y1.size()).zero();
  cxxblas::hemv(a1.size(1), a1.data(), x1.data(), v1.data(), alpha, beta,
                a1.stride(0), x1.stride(0), v1.stride(0));
  for (typename T::reference_iterator begin = v1.reference_begin(),
           end = v1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(y1(begin.position()), *begin);
  }

  T a2 = uniformDist(T(3, 9, 7, 7));
  T x2 = uniformDist(T(3, 9, 7));
  T y2 = linalg.hemv(a2, x2, alpha, beta);
  EXPECT_EQ(a2.dimension() - 1, y2.dimension());
  for (typename T::dim_type i = 0; i < y2.dimension(); ++i) {
    EXPECT_EQ(a2.size(i), y2.size(i));
  }
  T v2 = T(y2.size()).zero();
  T x2_select = x2.select(x2.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x2_select.reference_begin(),
           end = x2_select.reference_end(); begin != end; ++begin) {
    cxxblas::hemv(
        a2.size(a2.dimension() - 1), a2[begin.position()].data(),
        x2[begin.position()].data(), v2[begin.position()].data(), alpha, beta,
        a2.stride(a2.dimension() - 2), x2.stride(x2.dimension() - 1),
        v2.stride(v2.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v2.reference_begin(),
           end = v2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(y2(begin.position()), *begin);
  }

  T a3 = uniformDist(T({3, 9, 7, 7}, {1335, 148, 21, 1}));
  T x3 = uniformDist(T({3, 9, 7}, {193, 21, 2}));
  T y3 = linalg.hemv(a3, x3, alpha, beta);
  EXPECT_EQ(a3.dimension() - 1, y3.dimension());
  for (typename T::dim_type i = 0; i < y3.dimension(); ++i) {
    EXPECT_EQ(a3.size(i), y3.size(i));
  }
  T v3 = T(y3.size()).zero();
  T x3_select = x3.select(x3.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x3_select.reference_begin(),
           end = x3_select.reference_end(); begin != end; ++begin) {
    cxxblas::hemv(
        a3.size(a3.dimension() - 1), a3[begin.position()].data(),
        x3[begin.position()].data(), v3[begin.position()].data(), alpha, beta,
        a3.stride(a3.dimension() - 2), x3.stride(x3.dimension() - 1),
        v3.stride(v3.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v3.reference_begin(),
           end = v3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(y3(begin.position()), *begin);
  }
}

TEST(LinalgTest, hemvTest) {
  hemvTest< DoubleLinalg >();
  hemvTest< FloatLinalg >();
  hemvTest< DoubleComplexLinalg >();
  hemvTest< FloatComplexLinalg >();
}

template < typename L >
void herTest() {
  typedef typename L::tensor_type T;
  typedef typename T::real_type R;

  L linalg;
  R alpha = R(1.57);

  T x1 = uniformDist(T(7));
  T a1 = linalg.her(x1, alpha);
  EXPECT_EQ(x1.dimension() + 1, a1.dimension());
  T v1 = T(a1.size()).zero();
  cxxblas::her(x1.size(0), x1.data(), v1.data(), alpha, x1.stride(0),
               v1.stride(0));
  for (typename T::reference_iterator begin = v1.reference_begin(),
           end = v1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a1(begin.position()), *begin);
  }

  T x2 = uniformDist(T(3, 9, 7));
  T a2 = linalg.her(x2, alpha);
  EXPECT_EQ(x2.dimension() + 1, a2.dimension());
  for (typename T::dim_type i = 0; i < x2.dimension(); ++i) {
    EXPECT_EQ(x2.size(i), a2.size(i));
  }
  EXPECT_EQ(x2.size(x2.dimension() - 1), a2.size(a2.dimension() - 1));
  T v2 = T(a2.size()).zero();
  T x2_select = x2.select(x2.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x2_select.reference_begin(),
           end = x2_select.reference_end(); begin != end; ++begin) {
    cxxblas::her(
        x2.size(x2.dimension() - 1), x2[begin.position()].data(),
        v2[begin.position()].data(), alpha, x2.stride(x2.dimension() - 1),
        v2.stride(v2.dimension() - 2));
  }
  for (typename T::reference_iterator begin = v2.reference_begin(),
           end = v2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a2(begin.position()), *begin);
  }

  T x3 = uniformDist(T({3, 9, 7}, {47, 15, 2}));
  T a3 = linalg.her(x3, alpha);
  EXPECT_EQ(x3.dimension() + 1, a3.dimension());
  for (typename T::dim_type i = 0; i < x3.dimension(); ++i) {
    EXPECT_EQ(x3.size(i), a3.size(i));
  }
  EXPECT_EQ(x3.size(x3.dimension() - 1), a3.size(a3.dimension() - 1));
  T v3 = T(a3.size()).zero();
  T x3_select = x3.select(x3.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x3_select.reference_begin(),
           end = x3_select.reference_end(); begin != end; ++begin) {
    cxxblas::her(
        x3.size(x3.dimension() - 1), x3[begin.position()].data(),
        v3[begin.position()].data(), alpha, x3.stride(x3.dimension() - 1),
        v3.stride(v3.dimension() - 2));
  }
  for (typename T::reference_iterator begin = v3.reference_begin(),
           end = v3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a3(begin.position()), *begin);
  }
}

TEST(LinalgTest, herTest) {
  herTest< DoubleLinalg >();
  herTest< FloatLinalg >();
  herTest< DoubleComplexLinalg >();
  herTest< FloatComplexLinalg >();
}

template < typename L >
void her2Test() {
  typedef typename L::tensor_type T;
  typedef typename T::value_type D;

  L linalg;
  D alpha = D(1.57);

  T x1 = uniformDist(T(7));
  T y1 = uniformDist(T(7));
  T a1 = linalg.her2(x1, y1, alpha);
  EXPECT_EQ(x1.dimension() + 1, a1.dimension());
  T v1 = T(a1.size()).zero();
  cxxblas::her2(x1.size(0), x1.data(), y1.data(), v1.data(), alpha,
                x1.stride(0), y1.stride(0), v1.stride(0));
  for (typename T::reference_iterator begin = v1.reference_begin(),
           end = v1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a1(begin.position()), *begin);
  }

  T x2 = uniformDist(T(3, 9, 7));
  T y2 = uniformDist(T(3, 9, 7));
  T a2 = linalg.her2(x2, y2, alpha);
  EXPECT_EQ(x2.dimension() + 1, a2.dimension());
  for (typename T::dim_type i = 0; i < x2.dimension(); ++i) {
    EXPECT_EQ(x2.size(i), a2.size(i));
  }
  EXPECT_EQ(y2.size(y2.dimension() - 1), a2.size(a2.dimension() - 1));
  T v2 = T(a2.size()).zero();
  T x2_select = x2.select(x2.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x2_select.reference_begin(),
           end = x2_select.reference_end(); begin != end; ++begin) {
    cxxblas::her2(
        x2.size(x2.dimension() - 1), x2[begin.position()].data(),
        y2[begin.position()].data(), v2[begin.position()].data(), alpha,
        x2.stride(x2.dimension() - 1), y2.stride(y2.dimension() - 1),
        v2.stride(v2.dimension() - 2));
  }
  for (typename T::reference_iterator begin = v2.reference_begin(),
           end = v2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a2(begin.position()), *begin);
  }

  T x3 = uniformDist(T({3, 9, 7}, {47, 15, 2}));
  T y3 = uniformDist(T({3, 9, 7}, {69, 22, 2}));
  T a3 = linalg.her2(x3, y3, alpha);
  EXPECT_EQ(x3.dimension() + 1, a3.dimension());
  for (typename T::dim_type i = 0; i < x3.dimension(); ++i) {
    EXPECT_EQ(x3.size(i), a3.size(i));
  }
  EXPECT_EQ(y3.size(y3.dimension() - 1), a3.size(a3.dimension() - 1));
  T v3 = T(a3.size()).zero();
  T x3_select = x3.select(x3.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x3_select.reference_begin(),
           end = x3_select.reference_end(); begin != end; ++begin) {
    cxxblas::her2(
        x3.size(x3.dimension() - 1), x3[begin.position()].data(),
        y3[begin.position()].data(), v3[begin.position()].data(), alpha,
        x3.stride(x3.dimension() - 1), y3.stride(y3.dimension() - 1),
        v3.stride(v3.dimension() - 2));
  }
  for (typename T::reference_iterator begin = v3.reference_begin(),
           end = v3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a3(begin.position()), *begin);
  }
}

TEST(LinalgTest, her2Test) {
  her2Test< DoubleLinalg >();
  her2Test< FloatLinalg >();
  her2Test< DoubleComplexLinalg >();
  her2Test< FloatComplexLinalg >();
}

template < typename L >
void hpmvTest() {
  typedef typename L::tensor_type T;
  typedef typename T::value_type D;

  L linalg;
  D alpha = D(1.57);
  D beta = D(1.36);

  T a1 = uniformDist(T(7 * 8 / 2));
  T x1 = uniformDist(T(7));
  T y1 = linalg.hpmv(a1, x1, alpha, beta);
  EXPECT_EQ(a1.dimension(), y1.dimension());
  EXPECT_EQ(a1.size(0), y1.size(0) * (y1.size(0) + 1) / 2);
  T v1 = T(y1.size()).zero();
  cxxblas::hpmv(x1.size(0), a1.data(), x1.data(), v1.data(), alpha, beta,
                x1.stride(0), v1.stride(0));
  for (typename T::reference_iterator begin = v1.reference_begin(),
           end = v1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(y1(begin.position()), *begin);
  }

  T a2 = uniformDist(T(3, 9, 7 * 8 / 2));
  T x2 = uniformDist(T(3, 9, 7));
  T y2 = linalg.hpmv(a2, x2, alpha, beta);
  EXPECT_EQ(a2.dimension(), y2.dimension());
  for (typename T::dim_type i = 0; i < y2.dimension() - 1; ++i) {
    EXPECT_EQ(a2.size(i), y2.size(i));
  }
  EXPECT_EQ(
      a2.size(a2.dimension() - 1),
      y2.size(y2.dimension() - 1) * (y2.size(y2.dimension() - 1) + 1) / 2);
  T v2 = T(y2.size()).zero();
  T x2_select = x2.select(x2.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x2_select.reference_begin(),
           end = x2_select.reference_end(); begin != end; ++begin) {
    cxxblas::hpmv(
        x2.size(x2.dimension() - 1), a2[begin.position()].data(),
        x2[begin.position()].data(), v2[begin.position()].data(), alpha, beta,
        x2.stride(x2.dimension() - 1), v2.stride(v2.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v2.reference_begin(),
           end = v2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(y2(begin.position()), *begin);
  }

  T a3 = uniformDist(T({3, 9, 7 * 8 / 2}, {1335, 148, 1}));
  T x3 = uniformDist(T({3, 9, 7}, {193, 21, 2}));
  T y3 = linalg.hpmv(a3, x3, alpha, beta);
  EXPECT_EQ(a3.dimension(), y3.dimension());
  for (typename T::dim_type i = 0; i < y3.dimension() - 1; ++i) {
    EXPECT_EQ(a3.size(i), y3.size(i));
  }
  EXPECT_EQ(
      a3.size(a3.dimension() - 1),
      y3.size(y3.dimension() - 1) * (y3.size(y3.dimension() - 1) + 1) / 2);
  T v3 = T(y3.size()).zero();
  T x3_select = x3.select(x3.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x3_select.reference_begin(),
           end = x3_select.reference_end(); begin != end; ++begin) {
    cxxblas::hpmv(
        x3.size(x3.dimension() - 1), a3[begin.position()].data(),
        x3[begin.position()].data(), v3[begin.position()].data(), alpha, beta,
        x3.stride(x3.dimension() - 1), v3.stride(v3.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v3.reference_begin(),
           end = v3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(y3(begin.position()), *begin);
  }
}

TEST(LinalgTest, hpmvTest) {
  hpmvTest< DoubleLinalg >();
  hpmvTest< FloatLinalg >();
  hpmvTest< DoubleComplexLinalg >();
  hpmvTest< FloatComplexLinalg >();
}

template < typename L >
void hprTest() {
  typedef typename L::tensor_type T;
  typedef typename T::real_type R;

  L linalg;
  R alpha = R(1.57);

  T x1 = uniformDist(T(7));
  T a1 = linalg.hpr(x1, alpha);
  EXPECT_EQ(x1.dimension(), a1.dimension());
  EXPECT_EQ(x1.size(0) * (x1.size(0) + 1) / 2, a1.size(0));
  T v1 = T(a1.size()).zero();
  cxxblas::hpr(x1.size(0), x1.data(), v1.data(), alpha, x1.stride(0));
  for (typename T::reference_iterator begin = v1.reference_begin(),
           end = v1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a1(begin.position()), *begin);
  }

  T x2 = uniformDist(T(3, 9, 7));
  T a2 = linalg.hpr(x2, alpha);
  EXPECT_EQ(x2.dimension(), a2.dimension());
  for (typename T::dim_type i = 0; i < x2.dimension() - 1; ++i) {
    EXPECT_EQ(x2.size(i), a2.size(i));
  }
  EXPECT_EQ(
      x2.size(x2.dimension() - 1) * (x2.size(x2.dimension() - 1) + 1) / 2,
      a2.size(a2.dimension() - 1));
  T v2 = T(a2.size()).zero();
  T x2_select = x2.select(x2.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x2_select.reference_begin(),
           end = x2_select.reference_end(); begin != end; ++begin) {
    cxxblas::hpr(
        x2.size(x2.dimension() - 1), x2[begin.position()].data(),
        v2[begin.position()].data(), alpha, x2.stride(x2.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v2.reference_begin(),
           end = v2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a2(begin.position()), *begin);
  }

  T x3 = uniformDist(T({3, 9, 7}, {47, 15, 2}));
  T a3 = linalg.hpr(x3, alpha);
  EXPECT_EQ(x3.dimension(), a3.dimension());
  for (typename T::dim_type i = 0; i < x3.dimension() - 1; ++i) {
    EXPECT_EQ(x3.size(i), a3.size(i));
  }
  EXPECT_EQ(
      x3.size(x3.dimension() - 1) * (x3.size(x3.dimension() - 1) + 1) / 2,
      a3.size(a3.dimension() - 1));
  T v3 = T(a3.size()).zero();
  T x3_select = x3.select(x3.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x3_select.reference_begin(),
           end = x3_select.reference_end(); begin != end; ++begin) {
    cxxblas::hpr(
        x3.size(x3.dimension() - 1), x3[begin.position()].data(),
        v3[begin.position()].data(), alpha, x3.stride(x3.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v3.reference_begin(),
           end = v3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a3(begin.position()), *begin);
  }
}

TEST(LinalgTest, hprTest) {
  hprTest< DoubleLinalg >();
  hprTest< FloatLinalg >();
  hprTest< DoubleComplexLinalg >();
  hprTest< FloatComplexLinalg >();
}

template < typename L >
void hpr2Test() {
  typedef typename L::tensor_type T;
  typedef typename T::value_type D;

  L linalg;
  D alpha = D(1.57);

  T x1 = uniformDist(T(7));
  T y1 = uniformDist(T(7));
  T a1 = linalg.hpr2(x1, y1, alpha);
  EXPECT_EQ(x1.dimension(), a1.dimension());
  EXPECT_EQ(
      x1.size(x1.dimension() - 1) * (x1.size(x1.dimension() - 1) + 1) / 2,
      a1.size(a1.dimension() - 1));
  T v1 = T(a1.size()).zero();
  cxxblas::hpr2(x1.size(0), x1.data(), y1.data(), v1.data(), alpha,
                x1.stride(0), y1.stride(0));
  for (typename T::reference_iterator begin = v1.reference_begin(),
           end = v1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a1(begin.position()), *begin);
  }

  T x2 = uniformDist(T(3, 9, 7));
  T y2 = uniformDist(T(3, 9, 7));
  T a2 = linalg.hpr2(x2, y2, alpha);
  EXPECT_EQ(x2.dimension(), a2.dimension());
  for (typename T::dim_type i = 0; i < x2.dimension() - 1; ++i) {
    EXPECT_EQ(x2.size(i), a2.size(i));
  }
  EXPECT_EQ(
      x2.size(x2.dimension() - 1) * (x2.size(x2.dimension() - 1) + 1) / 2,
      a2.size(a2.dimension() - 1));
  T v2 = T(a2.size()).zero();
  T x2_select = x2.select(x2.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x2_select.reference_begin(),
           end = x2_select.reference_end(); begin != end; ++begin) {
    cxxblas::hpr2(
        x2.size(x2.dimension() - 1), x2[begin.position()].data(),
        y2[begin.position()].data(), v2[begin.position()].data(), alpha,
        x2.stride(x2.dimension() - 1), y2.stride(y2.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v2.reference_begin(),
           end = v2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a2(begin.position()), *begin);
  }

  T x3 = uniformDist(T({3, 9, 7}, {47, 15, 2}));
  T y3 = uniformDist(T({3, 9, 7}, {69, 22, 2}));
  T a3 = linalg.hpr2(x3, y3, alpha);
  EXPECT_EQ(x3.dimension(), a3.dimension());
  for (typename T::dim_type i = 0; i < x3.dimension() - 1; ++i) {
    EXPECT_EQ(x3.size(i), a3.size(i));
  }
  EXPECT_EQ(
      x3.size(x3.dimension() - 1) * (x3.size(x3.dimension() - 1) + 1) / 2,
      a3.size(a3.dimension() - 1));
  T v3 = T(a3.size()).zero();
  T x3_select = x3.select(x3.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x3_select.reference_begin(),
           end = x3_select.reference_end(); begin != end; ++begin) {
    cxxblas::hpr2(
        x3.size(x3.dimension() - 1), x3[begin.position()].data(),
        y3[begin.position()].data(), v3[begin.position()].data(), alpha,
        x3.stride(x3.dimension() - 1), y3.stride(y3.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v3.reference_begin(),
           end = v3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a3(begin.position()), *begin);
  }
}

TEST(LinalgTest, hpr2Test) {
  hpr2Test< DoubleLinalg >();
  hpr2Test< FloatLinalg >();
  hpr2Test< DoubleComplexLinalg >();
  hpr2Test< FloatComplexLinalg >();
}

template < typename L >
void sbmvTest() {
  typedef typename L::tensor_type T;
  typedef typename T::value_type D;

  L linalg;
  D alpha = D(1.57);
  D beta = D(1.36);
  typename T::size_type k = 2;

  T a1 = uniformDist(T(7, k + 1));
  T x1 = uniformDist(T(7));
  T y1 = linalg.sbmv(a1, x1, alpha, beta, k);
  EXPECT_EQ(a1.dimension() - 1, y1.dimension());
  EXPECT_EQ(a1.size(0), y1.size(0));
  T v1 = T(y1.size()).zero();
  cxxblas::sbmv(x1.size(0), a1.data(), x1.data(), v1.data(), alpha,
                beta, k, a1.stride(0), x1.stride(0), v1.stride(0));
  for (typename T::reference_iterator begin = v1.reference_begin(),
           end = v1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(y1(begin.position()), *begin);
  }

  T a2 = uniformDist(T(3, 9, 7, k + 1));
  T x2 = uniformDist(T(3, 9, 7));
  T y2 = linalg.sbmv(a2, x2, alpha, beta, k);
  EXPECT_EQ(a2.dimension() - 1, y2.dimension());
  for (typename T::dim_type i = 0; i < y2.dimension(); ++i) {
    EXPECT_EQ(a2.size(i), y2.size(i));
  }
  T v2 = T(y2.size()).zero();
  T x2_select = x2.select(x2.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x2_select.reference_begin(),
           end = x2_select.reference_end(); begin != end; ++begin) {
    cxxblas::sbmv(
        x2.size(x2.dimension() - 1), a2[begin.position()].data(),
        x2[begin.position()].data(), v2[begin.position()].data(), alpha, beta,
        k, a2.stride(a2.dimension() - 2), x2.stride(x2.dimension() - 1),
        v2.stride(v2.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v2.reference_begin(),
           end = v2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(y2(begin.position()), *begin);
  }

  T a3 = uniformDist(T({3, 9, 7, k + 1}, {1335, 148, 21, 1}));
  T x3 = uniformDist(T({3, 9, 7}, {193, 21, 2}));
  T y3 = linalg.sbmv(a3, x3, alpha, beta, k);
  EXPECT_EQ(a3.dimension() - 1, y3.dimension());
  for (typename T::dim_type i = 0; i < y3.dimension(); ++i) {
    EXPECT_EQ(a3.size(i), y3.size(i));
  }
  T v3 = T(y3.size()).zero();
  T x3_select = x3.select(x3.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x3_select.reference_begin(),
           end = x3_select.reference_end(); begin != end; ++begin) {
    cxxblas::sbmv(
        x3.size(x3.dimension() - 1), a3[begin.position()].data(),
        x3[begin.position()].data(), v3[begin.position()].data(), alpha, beta,
        k, a3.stride(a3.dimension() - 2), x3.stride(x3.dimension() - 1),
        v3.stride(v3.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v3.reference_begin(),
           end = v3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(y3(begin.position()), *begin);
  }
}

TEST(LinalgTest, sbmvTest) {
  sbmvTest< DoubleLinalg >();
  sbmvTest< FloatLinalg >();
  sbmvTest< DoubleComplexLinalg >();
  sbmvTest< FloatComplexLinalg >();
}

template < typename L >
void spmvTest() {
  typedef typename L::tensor_type T;
  typedef typename T::value_type D;

  L linalg;
  D alpha = D(1.57);
  D beta = D(1.36);

  T a1 = uniformDist(T(7 * 8 / 2));
  T x1 = uniformDist(T(7));
  T y1 = linalg.spmv(a1, x1, alpha, beta);
  EXPECT_EQ(a1.dimension(), y1.dimension());
  EXPECT_EQ(a1.size(0), y1.size(0) * (y1.size(0) + 1) / 2);
  T v1 = T(y1.size()).zero();
  cxxblas::spmv(x1.size(0), a1.data(), x1.data(), v1.data(), alpha, beta,
                x1.stride(0), v1.stride(0));
  for (typename T::reference_iterator begin = v1.reference_begin(),
           end = v1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(y1(begin.position()), *begin);
  }

  T a2 = uniformDist(T(3, 9, 7 * 8 / 2));
  T x2 = uniformDist(T(3, 9, 7));
  T y2 = linalg.spmv(a2, x2, alpha, beta);
  EXPECT_EQ(a2.dimension(), y2.dimension());
  for (typename T::dim_type i = 0; i < y2.dimension() - 1; ++i) {
    EXPECT_EQ(a2.size(i), y2.size(i));
  }
  EXPECT_EQ(
      a2.size(a2.dimension() - 1),
      y2.size(y2.dimension() - 1) * (y2.size(y2.dimension() - 1) + 1) / 2);
  T v2 = T(y2.size()).zero();
  T x2_select = x2.select(x2.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x2_select.reference_begin(),
           end = x2_select.reference_end(); begin != end; ++begin) {
    cxxblas::spmv(
        x2.size(x2.dimension() - 1), a2[begin.position()].data(),
        x2[begin.position()].data(), v2[begin.position()].data(), alpha, beta,
        x2.stride(x2.dimension() - 1), v2.stride(v2.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v2.reference_begin(),
           end = v2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(y2(begin.position()), *begin);
  }

  T a3 = uniformDist(T({3, 9, 7 * 8 / 2}, {1335, 148, 1}));
  T x3 = uniformDist(T({3, 9, 7}, {193, 21, 2}));
  T y3 = linalg.spmv(a3, x3, alpha, beta);
  EXPECT_EQ(a3.dimension(), y3.dimension());
  for (typename T::dim_type i = 0; i < y3.dimension() - 1; ++i) {
    EXPECT_EQ(a3.size(i), y3.size(i));
  }
  EXPECT_EQ(
      a3.size(a3.dimension() - 1),
      y3.size(y3.dimension() - 1) * (y3.size(y3.dimension() - 1) + 1) / 2);
  T v3 = T(y3.size()).zero();
  T x3_select = x3.select(x3.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x3_select.reference_begin(),
           end = x3_select.reference_end(); begin != end; ++begin) {
    cxxblas::spmv(
        x3.size(x3.dimension() - 1), a3[begin.position()].data(),
        x3[begin.position()].data(), v3[begin.position()].data(), alpha, beta,
        x3.stride(x3.dimension() - 1), v3.stride(v3.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v3.reference_begin(),
           end = v3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(y3(begin.position()), *begin);
  }
}

TEST(LinalgTest, spmvTest) {
  spmvTest< DoubleLinalg >();
  spmvTest< FloatLinalg >();
  spmvTest< DoubleComplexLinalg >();
  spmvTest< FloatComplexLinalg >();
}

template < typename L >
void sprTest() {
  typedef typename L::tensor_type T;
  typedef typename T::real_type R;

  L linalg;
  R alpha = R(1.57);

  T x1 = uniformDist(T(7));
  T a1 = linalg.spr(x1, alpha);
  EXPECT_EQ(x1.dimension(), a1.dimension());
  EXPECT_EQ(x1.size(0) * (x1.size(0) + 1) / 2, a1.size(0));
  T v1 = T(a1.size()).zero();
  cxxblas::spr(x1.size(0), x1.data(), v1.data(), alpha, x1.stride(0));
  for (typename T::reference_iterator begin = v1.reference_begin(),
           end = v1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a1(begin.position()), *begin);
  }

  T x2 = uniformDist(T(3, 9, 7));
  T a2 = linalg.spr(x2, alpha);
  EXPECT_EQ(x2.dimension(), a2.dimension());
  for (typename T::dim_type i = 0; i < x2.dimension() - 1; ++i) {
    EXPECT_EQ(x2.size(i), a2.size(i));
  }
  EXPECT_EQ(
      x2.size(x2.dimension() - 1) * (x2.size(x2.dimension() - 1) + 1) / 2,
      a2.size(a2.dimension() - 1));
  T v2 = T(a2.size()).zero();
  T x2_select = x2.select(x2.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x2_select.reference_begin(),
           end = x2_select.reference_end(); begin != end; ++begin) {
    cxxblas::spr(
        x2.size(x2.dimension() - 1), x2[begin.position()].data(),
        v2[begin.position()].data(), alpha, x2.stride(x2.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v2.reference_begin(),
           end = v2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a2(begin.position()), *begin);
  }

  T x3 = uniformDist(T({3, 9, 7}, {47, 15, 2}));
  T a3 = linalg.spr(x3, alpha);
  EXPECT_EQ(x3.dimension(), a3.dimension());
  for (typename T::dim_type i = 0; i < x3.dimension() - 1; ++i) {
    EXPECT_EQ(x3.size(i), a3.size(i));
  }
  EXPECT_EQ(
      x3.size(x3.dimension() - 1) * (x3.size(x3.dimension() - 1) + 1) / 2,
      a3.size(a3.dimension() - 1));
  T v3 = T(a3.size()).zero();
  T x3_select = x3.select(x3.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x3_select.reference_begin(),
           end = x3_select.reference_end(); begin != end; ++begin) {
    cxxblas::spr(
        x3.size(x3.dimension() - 1), x3[begin.position()].data(),
        v3[begin.position()].data(), alpha, x3.stride(x3.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v3.reference_begin(),
           end = v3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a3(begin.position()), *begin);
  }
}

TEST(LinalgTest, sprTest) {
  sprTest< DoubleLinalg >();
  sprTest< FloatLinalg >();
  sprTest< DoubleComplexLinalg >();
  sprTest< FloatComplexLinalg >();
}

template < typename L >
void spr2Test() {
  typedef typename L::tensor_type T;
  typedef typename T::value_type D;

  L linalg;
  D alpha = D(1.57);

  T x1 = uniformDist(T(7));
  T y1 = uniformDist(T(7));
  T a1 = linalg.spr2(x1, y1, alpha);
  EXPECT_EQ(x1.dimension(), a1.dimension());
  EXPECT_EQ(
      x1.size(x1.dimension() - 1) * (x1.size(x1.dimension() - 1) + 1) / 2,
      a1.size(a1.dimension() - 1));
  T v1 = T(a1.size()).zero();
  cxxblas::spr2(x1.size(0), x1.data(), y1.data(), v1.data(), alpha,
                x1.stride(0), y1.stride(0));
  for (typename T::reference_iterator begin = v1.reference_begin(),
           end = v1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a1(begin.position()), *begin);
  }

  T x2 = uniformDist(T(3, 9, 7));
  T y2 = uniformDist(T(3, 9, 7));
  T a2 = linalg.spr2(x2, y2, alpha);
  EXPECT_EQ(x2.dimension(), a2.dimension());
  for (typename T::dim_type i = 0; i < x2.dimension() - 1; ++i) {
    EXPECT_EQ(x2.size(i), a2.size(i));
  }
  EXPECT_EQ(
      x2.size(x2.dimension() - 1) * (x2.size(x2.dimension() - 1) + 1) / 2,
      a2.size(a2.dimension() - 1));
  T v2 = T(a2.size()).zero();
  T x2_select = x2.select(x2.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x2_select.reference_begin(),
           end = x2_select.reference_end(); begin != end; ++begin) {
    cxxblas::spr2(
        x2.size(x2.dimension() - 1), x2[begin.position()].data(),
        y2[begin.position()].data(), v2[begin.position()].data(), alpha,
        x2.stride(x2.dimension() - 1), y2.stride(y2.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v2.reference_begin(),
           end = v2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a2(begin.position()), *begin);
  }

  T x3 = uniformDist(T({3, 9, 7}, {47, 15, 2}));
  T y3 = uniformDist(T({3, 9, 7}, {69, 22, 2}));
  T a3 = linalg.spr2(x3, y3, alpha);
  EXPECT_EQ(x3.dimension(), a3.dimension());
  for (typename T::dim_type i = 0; i < x3.dimension() - 1; ++i) {
    EXPECT_EQ(x3.size(i), a3.size(i));
  }
  EXPECT_EQ(
      x3.size(x3.dimension() - 1) * (x3.size(x3.dimension() - 1) + 1) / 2,
      a3.size(a3.dimension() - 1));
  T v3 = T(a3.size()).zero();
  T x3_select = x3.select(x3.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x3_select.reference_begin(),
           end = x3_select.reference_end(); begin != end; ++begin) {
    cxxblas::spr2(
        x3.size(x3.dimension() - 1), x3[begin.position()].data(),
        y3[begin.position()].data(), v3[begin.position()].data(), alpha,
        x3.stride(x3.dimension() - 1), y3.stride(y3.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v3.reference_begin(),
           end = v3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a3(begin.position()), *begin);
  }
}

TEST(LinalgTest, spr2Test) {
  spr2Test< DoubleLinalg >();
  spr2Test< FloatLinalg >();
  spr2Test< DoubleComplexLinalg >();
  spr2Test< FloatComplexLinalg >();
}

template < typename L >
void symvTest() {
  typedef typename L::tensor_type T;
  typedef typename T::value_type D;

  L linalg;
  D alpha = D(1.57);
  D beta = D(1.36);

  T a1 = uniformDist(T(7, 7));
  T x1 = uniformDist(T(7));
  T y1 = linalg.symv(a1, x1, alpha, beta);
  EXPECT_EQ(a1.dimension() - 1, y1.dimension());
  EXPECT_EQ(a1.size(0), y1.size(0));
  T v1 = T(y1.size()).zero();
  cxxblas::symv(a1.size(1), a1.data(), x1.data(), v1.data(), alpha, beta,
                a1.stride(0), x1.stride(0), v1.stride(0));
  for (typename T::reference_iterator begin = v1.reference_begin(),
           end = v1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(y1(begin.position()), *begin);
  }

  T a2 = uniformDist(T(3, 9, 7, 7));
  T x2 = uniformDist(T(3, 9, 7));
  T y2 = linalg.symv(a2, x2, alpha, beta);
  EXPECT_EQ(a2.dimension() - 1, y2.dimension());
  for (typename T::dim_type i = 0; i < y2.dimension(); ++i) {
    EXPECT_EQ(a2.size(i), y2.size(i));
  }
  T v2 = T(y2.size()).zero();
  T x2_select = x2.select(x2.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x2_select.reference_begin(),
           end = x2_select.reference_end(); begin != end; ++begin) {
    cxxblas::symv(
        a2.size(a2.dimension() - 1), a2[begin.position()].data(),
        x2[begin.position()].data(), v2[begin.position()].data(), alpha, beta,
        a2.stride(a2.dimension() - 2), x2.stride(x2.dimension() - 1),
        v2.stride(v2.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v2.reference_begin(),
           end = v2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(y2(begin.position()), *begin);
  }

  T a3 = uniformDist(T({3, 9, 7, 7}, {1335, 148, 21, 1}));
  T x3 = uniformDist(T({3, 9, 7}, {193, 21, 2}));
  T y3 = linalg.symv(a3, x3, alpha, beta);
  EXPECT_EQ(a3.dimension() - 1, y3.dimension());
  for (typename T::dim_type i = 0; i < y3.dimension(); ++i) {
    EXPECT_EQ(a3.size(i), y3.size(i));
  }
  T v3 = T(y3.size()).zero();
  T x3_select = x3.select(x3.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x3_select.reference_begin(),
           end = x3_select.reference_end(); begin != end; ++begin) {
    cxxblas::symv(
        a3.size(a3.dimension() - 1), a3[begin.position()].data(),
        x3[begin.position()].data(), v3[begin.position()].data(), alpha, beta,
        a3.stride(a3.dimension() - 2), x3.stride(x3.dimension() - 1),
        v3.stride(v3.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v3.reference_begin(),
           end = v3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(y3(begin.position()), *begin);
  }
}

TEST(LinalgTest, symvTest) {
  symvTest< DoubleLinalg >();
  symvTest< FloatLinalg >();
  symvTest< DoubleComplexLinalg >();
  symvTest< FloatComplexLinalg >();
}

template < typename L >
void syrTest() {
  typedef typename L::tensor_type T;
  typedef typename T::real_type R;

  L linalg;
  R alpha = R(1.57);

  T x1 = uniformDist(T(7));
  T a1 = linalg.syr(x1, alpha);
  EXPECT_EQ(x1.dimension() + 1, a1.dimension());
  T v1 = T(a1.size()).zero();
  cxxblas::syr(x1.size(0), x1.data(), v1.data(), alpha, x1.stride(0),
               v1.stride(0));
  for (typename T::reference_iterator begin = v1.reference_begin(),
           end = v1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a1(begin.position()), *begin);
  }

  T x2 = uniformDist(T(3, 9, 7));
  T a2 = linalg.syr(x2, alpha);
  EXPECT_EQ(x2.dimension() + 1, a2.dimension());
  for (typename T::dim_type i = 0; i < x2.dimension(); ++i) {
    EXPECT_EQ(x2.size(i), a2.size(i));
  }
  EXPECT_EQ(x2.size(x2.dimension() - 1), a2.size(a2.dimension() - 1));
  T v2 = T(a2.size()).zero();
  T x2_select = x2.select(x2.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x2_select.reference_begin(),
           end = x2_select.reference_end(); begin != end; ++begin) {
    cxxblas::syr(
        x2.size(x2.dimension() - 1), x2[begin.position()].data(),
        v2[begin.position()].data(), alpha, x2.stride(x2.dimension() - 1),
        v2.stride(v2.dimension() - 2));
  }
  for (typename T::reference_iterator begin = v2.reference_begin(),
           end = v2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a2(begin.position()), *begin);
  }

  T x3 = uniformDist(T({3, 9, 7}, {47, 15, 2}));
  T a3 = linalg.syr(x3, alpha);
  EXPECT_EQ(x3.dimension() + 1, a3.dimension());
  for (typename T::dim_type i = 0; i < x3.dimension(); ++i) {
    EXPECT_EQ(x3.size(i), a3.size(i));
  }
  EXPECT_EQ(x3.size(x3.dimension() - 1), a3.size(a3.dimension() - 1));
  T v3 = T(a3.size()).zero();
  T x3_select = x3.select(x3.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x3_select.reference_begin(),
           end = x3_select.reference_end(); begin != end; ++begin) {
    cxxblas::syr(
        x3.size(x3.dimension() - 1), x3[begin.position()].data(),
        v3[begin.position()].data(), alpha, x3.stride(x3.dimension() - 1),
        v3.stride(v3.dimension() - 2));
  }
  for (typename T::reference_iterator begin = v3.reference_begin(),
           end = v3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a3(begin.position()), *begin);
  }
}

TEST(LinalgTest, syrTest) {
  syrTest< DoubleLinalg >();
  syrTest< FloatLinalg >();
  syrTest< DoubleComplexLinalg >();
  syrTest< FloatComplexLinalg >();
}

template < typename L >
void syr2Test() {
  typedef typename L::tensor_type T;
  typedef typename T::value_type D;

  L linalg;
  D alpha = D(1.57);

  T x1 = uniformDist(T(7));
  T y1 = uniformDist(T(7));
  T a1 = linalg.syr2(x1, y1, alpha);
  EXPECT_EQ(x1.dimension() + 1, a1.dimension());
  T v1 = T(a1.size()).zero();
  cxxblas::syr2(x1.size(0), x1.data(), y1.data(), v1.data(), alpha,
                x1.stride(0), y1.stride(0), v1.stride(0));
  for (typename T::reference_iterator begin = v1.reference_begin(),
           end = v1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a1(begin.position()), *begin);
  }

  T x2 = uniformDist(T(3, 9, 7));
  T y2 = uniformDist(T(3, 9, 7));
  T a2 = linalg.syr2(x2, y2, alpha);
  EXPECT_EQ(x2.dimension() + 1, a2.dimension());
  for (typename T::dim_type i = 0; i < x2.dimension(); ++i) {
    EXPECT_EQ(x2.size(i), a2.size(i));
  }
  EXPECT_EQ(y2.size(y2.dimension() - 1), a2.size(a2.dimension() - 1));
  T v2 = T(a2.size()).zero();
  T x2_select = x2.select(x2.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x2_select.reference_begin(),
           end = x2_select.reference_end(); begin != end; ++begin) {
    cxxblas::syr2(
        x2.size(x2.dimension() - 1), x2[begin.position()].data(),
        y2[begin.position()].data(), v2[begin.position()].data(), alpha,
        x2.stride(x2.dimension() - 1), y2.stride(y2.dimension() - 1),
        v2.stride(v2.dimension() - 2));
  }
  for (typename T::reference_iterator begin = v2.reference_begin(),
           end = v2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a2(begin.position()), *begin);
  }

  T x3 = uniformDist(T({3, 9, 7}, {47, 15, 2}));
  T y3 = uniformDist(T({3, 9, 7}, {69, 22, 2}));
  T a3 = linalg.syr2(x3, y3, alpha);
  EXPECT_EQ(x3.dimension() + 1, a3.dimension());
  for (typename T::dim_type i = 0; i < x3.dimension(); ++i) {
    EXPECT_EQ(x3.size(i), a3.size(i));
  }
  EXPECT_EQ(y3.size(y3.dimension() - 1), a3.size(a3.dimension() - 1));
  T v3 = T(a3.size()).zero();
  T x3_select = x3.select(x3.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x3_select.reference_begin(),
           end = x3_select.reference_end(); begin != end; ++begin) {
    cxxblas::syr2(
        x3.size(x3.dimension() - 1), x3[begin.position()].data(),
        y3[begin.position()].data(), v3[begin.position()].data(), alpha,
        x3.stride(x3.dimension() - 1), y3.stride(y3.dimension() - 1),
        v3.stride(v3.dimension() - 2));
  }
  for (typename T::reference_iterator begin = v3.reference_begin(),
           end = v3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(a3(begin.position()), *begin);
  }
}

TEST(LinalgTest, syr2Test) {
  syr2Test< DoubleLinalg >();
  syr2Test< FloatLinalg >();
  syr2Test< DoubleComplexLinalg >();
  syr2Test< FloatComplexLinalg >();
}

template < typename L >
void tbmvTest() {
  typedef typename L::tensor_type T;

  L linalg;
  typename T::size_type k = 2;

  T a1 = uniformDist(T(7, k + 1));
  T x1 = linalg.tbmv(a1, k);
  EXPECT_EQ(a1.dimension() - 1, x1.dimension());
  EXPECT_EQ(a1.size(0), x1.size(0));
  T v1 = T(x1.size()).zero();
  cxxblas::tbmv(x1.size(0), a1.data(), v1.data(), k, a1.stride(0),
                v1.stride(0));
  for (typename T::reference_iterator begin = v1.reference_begin(),
           end = v1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(x1(begin.position()), *begin);
  }

  T a2 = uniformDist(T(3, 9, 7, k + 1));
  T x2 = linalg.tbmv(a2, k);
  EXPECT_EQ(a2.dimension() - 1, x2.dimension());
  for (typename T::dim_type i = 0; i < x2.dimension(); ++i) {
    EXPECT_EQ(a2.size(i), x2.size(i));
  }
  T v2 = T(x2.size()).zero();
  T x2_select = x2.select(x2.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x2_select.reference_begin(),
           end = x2_select.reference_end(); begin != end; ++begin) {
    cxxblas::tbmv(
        x2.size(x2.dimension() - 1), a2[begin.position()].data(),
        v2[begin.position()].data(), k, a2.stride(a2.dimension() - 2),
        v2.stride(v2.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v2.reference_begin(),
           end = v2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(x2(begin.position()), *begin);
  }

  T a3 = uniformDist(T({3, 9, 7, k + 1}, {1335, 148, 21, 1}));
  T x3 = linalg.tbmv(a3, k);
  EXPECT_EQ(a3.dimension() - 1, x3.dimension());
  for (typename T::dim_type i = 0; i < x3.dimension(); ++i) {
    EXPECT_EQ(a3.size(i), x3.size(i));
  }
  T v3 = T(x3.size()).zero();
  T x3_select = x3.select(x3.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x3_select.reference_begin(),
           end = x3_select.reference_end(); begin != end; ++begin) {
    cxxblas::tbmv(
        x3.size(x3.dimension() - 1), a3[begin.position()].data(),
        v3[begin.position()].data(), k, a3.stride(a3.dimension() - 2),
        v3.stride(v3.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v3.reference_begin(),
           end = v3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(x3(begin.position()), *begin);
  }
}

TEST(LinalgTest, tbmvTest) {
  tbmvTest< DoubleLinalg >();
  tbmvTest< FloatLinalg >();
  tbmvTest< DoubleComplexLinalg >();
  tbmvTest< FloatComplexLinalg >();
}

template < typename L >
void tbsvTest() {
  typedef typename L::tensor_type T;

  L linalg;
  typename T::size_type k = 2;

  T a1 = uniformDist(T(7, k + 1));
  T x1 = linalg.tbsv(a1, k);
  EXPECT_EQ(a1.dimension() - 1, x1.dimension());
  EXPECT_EQ(a1.size(0), x1.size(0));
  T v1 = T(x1.size()).zero();
  cxxblas::tbsv(x1.size(0), a1.data(), v1.data(), k, a1.stride(0),
                v1.stride(0));
  for (typename T::reference_iterator begin = v1.reference_begin(),
           end = v1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(x1(begin.position()), *begin);
  }

  T a2 = uniformDist(T(3, 9, 7, k + 1));
  T x2 = linalg.tbsv(a2, k);
  EXPECT_EQ(a2.dimension() - 1, x2.dimension());
  for (typename T::dim_type i = 0; i < x2.dimension(); ++i) {
    EXPECT_EQ(a2.size(i), x2.size(i));
  }
  T v2 = T(x2.size()).zero();
  T x2_select = x2.select(x2.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x2_select.reference_begin(),
           end = x2_select.reference_end(); begin != end; ++begin) {
    cxxblas::tbsv(
        x2.size(x2.dimension() - 1), a2[begin.position()].data(),
        v2[begin.position()].data(), k, a2.stride(a2.dimension() - 2),
        v2.stride(v2.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v2.reference_begin(),
           end = v2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(x2(begin.position()), *begin);
  }

  T a3 = uniformDist(T({3, 9, 7, k + 1}, {1335, 148, 21, 1}));
  T x3 = linalg.tbsv(a3, k);
  EXPECT_EQ(a3.dimension() - 1, x3.dimension());
  for (typename T::dim_type i = 0; i < x3.dimension(); ++i) {
    EXPECT_EQ(a3.size(i), x3.size(i));
  }
  T v3 = T(x3.size()).zero();
  T x3_select = x3.select(x3.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x3_select.reference_begin(),
           end = x3_select.reference_end(); begin != end; ++begin) {
    cxxblas::tbsv(
        x3.size(x3.dimension() - 1), a3[begin.position()].data(),
        v3[begin.position()].data(), k, a3.stride(a3.dimension() - 2),
        v3.stride(v3.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v3.reference_begin(),
           end = v3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(x3(begin.position()), *begin);
  }
}

TEST(LinalgTest, tbsvTest) {
  tbsvTest< DoubleLinalg >();
  tbsvTest< FloatLinalg >();
  tbsvTest< DoubleComplexLinalg >();
  tbsvTest< FloatComplexLinalg >();
}

template < typename L >
void tpmvTest() {
  typedef typename L::tensor_type T;

  L linalg;

  T a1 = uniformDist(T(7 * 8 / 2));
  T x1 = linalg.tpmv(a1);
  EXPECT_EQ(a1.dimension(), x1.dimension());
  EXPECT_EQ(a1.size(0), x1.size(0) * (x1.size(0) + 1) / 2);
  T v1 = T(x1.size()).zero();
  cxxblas::tpmv(x1.size(0), a1.data(), v1.data(), v1.stride(0));
  for (typename T::reference_iterator begin = v1.reference_begin(),
           end = v1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(x1(begin.position()), *begin);
  }

  T a2 = uniformDist(T(3, 9, 7 * 8 / 2));
  T x2 = linalg.tpmv(a2);
  EXPECT_EQ(a2.dimension(), x2.dimension());
  for (typename T::dim_type i = 0; i < x2.dimension() - 1; ++i) {
    EXPECT_EQ(a2.size(i), x2.size(i));
  }
  EXPECT_EQ(
      a2.size(a2.dimension() - 1),
      x2.size(x2.dimension() - 1) * (x2.size(x2.dimension() - 1) + 1) / 2);
  T v2 = T(x2.size()).zero();
  T x2_select = x2.select(x2.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x2_select.reference_begin(),
           end = x2_select.reference_end(); begin != end; ++begin) {
    cxxblas::tpmv(
        x2.size(x2.dimension() - 1), a2[begin.position()].data(),
        v2[begin.position()].data(), v2.stride(v2.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v2.reference_begin(),
           end = v2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(x2(begin.position()), *begin);
  }

  T a3 = uniformDist(T({3, 9, 7 * 8 / 2}, {1335, 148, 1}));
  T x3 = linalg.tpmv(a3);
  EXPECT_EQ(a3.dimension(), x3.dimension());
  for (typename T::dim_type i = 0; i < x3.dimension() - 1; ++i) {
    EXPECT_EQ(a3.size(i), x3.size(i));
  }
  EXPECT_EQ(
      a3.size(a3.dimension() - 1),
      x3.size(x3.dimension() - 1) * (x3.size(x3.dimension() - 1) + 1) / 2);
  T v3 = T(x3.size()).zero();
  T x3_select = x3.select(x3.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x3_select.reference_begin(),
           end = x3_select.reference_end(); begin != end; ++begin) {
    cxxblas::tpmv(
        x3.size(x3.dimension() - 1), a3[begin.position()].data(),
        v3[begin.position()].data(), v3.stride(v3.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v3.reference_begin(),
           end = v3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(x3(begin.position()), *begin);
  }
}

TEST(LinalgTest, tpmvTest) {
  tpmvTest< DoubleLinalg >();
  tpmvTest< FloatLinalg >();
  tpmvTest< DoubleComplexLinalg >();
  tpmvTest< FloatComplexLinalg >();
}

template < typename L >
void tpsvTest() {
  typedef typename L::tensor_type T;

  L linalg;

  T a1 = uniformDist(T(7 * 8 / 2));
  T x1 = linalg.tpsv(a1);
  EXPECT_EQ(a1.dimension(), x1.dimension());
  EXPECT_EQ(a1.size(0), x1.size(0) * (x1.size(0) + 1) / 2);
  T v1 = T(x1.size()).zero();
  cxxblas::tpsv(x1.size(0), a1.data(), v1.data(), v1.stride(0));
  for (typename T::reference_iterator begin = v1.reference_begin(),
           end = v1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(x1(begin.position()), *begin);
  }

  T a2 = uniformDist(T(3, 9, 7 * 8 / 2));
  T x2 = linalg.tpsv(a2);
  EXPECT_EQ(a2.dimension(), x2.dimension());
  for (typename T::dim_type i = 0; i < x2.dimension() - 1; ++i) {
    EXPECT_EQ(a2.size(i), x2.size(i));
  }
  EXPECT_EQ(
      a2.size(a2.dimension() - 1),
      x2.size(x2.dimension() - 1) * (x2.size(x2.dimension() - 1) + 1) / 2);
  T v2 = T(x2.size()).zero();
  T x2_select = x2.select(x2.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x2_select.reference_begin(),
           end = x2_select.reference_end(); begin != end; ++begin) {
    cxxblas::tpsv(
        x2.size(x2.dimension() - 1), a2[begin.position()].data(),
        v2[begin.position()].data(), v2.stride(v2.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v2.reference_begin(),
           end = v2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(x2(begin.position()), *begin);
  }

  T a3 = uniformDist(T({3, 9, 7 * 8 / 2}, {1335, 148, 1}));
  T x3 = linalg.tpsv(a3);
  EXPECT_EQ(a3.dimension(), x3.dimension());
  for (typename T::dim_type i = 0; i < x3.dimension() - 1; ++i) {
    EXPECT_EQ(a3.size(i), x3.size(i));
  }
  EXPECT_EQ(
      a3.size(a3.dimension() - 1),
      x3.size(x3.dimension() - 1) * (x3.size(x3.dimension() - 1) + 1) / 2);
  T v3 = T(x3.size()).zero();
  T x3_select = x3.select(x3.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x3_select.reference_begin(),
           end = x3_select.reference_end(); begin != end; ++begin) {
    cxxblas::tpsv(
        x3.size(x3.dimension() - 1), a3[begin.position()].data(),
        v3[begin.position()].data(), v3.stride(v3.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v3.reference_begin(),
           end = v3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(x3(begin.position()), *begin);
  }
}

TEST(LinalgTest, tpsvTest) {
  tpsvTest< DoubleLinalg >();
  tpsvTest< FloatLinalg >();
  tpsvTest< DoubleComplexLinalg >();
  tpsvTest< FloatComplexLinalg >();
}

template < typename L >
void trmvTest() {
  typedef typename L::tensor_type T;

  L linalg;

  T a1 = uniformDist(T(7, 7));
  T x1 = linalg.trmv(a1);
  EXPECT_EQ(a1.dimension() - 1, x1.dimension());
  EXPECT_EQ(a1.size(0), x1.size(0));
  T v1 = T(x1.size()).zero();
  cxxblas::trmv(a1.size(1), a1.data(), v1.data(), a1.stride(0), v1.stride(0));
  for (typename T::reference_iterator begin = v1.reference_begin(),
           end = v1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(x1(begin.position()), *begin);
  }

  T a2 = uniformDist(T(3, 9, 7, 7));
  T x2 = linalg.trmv(a2);
  EXPECT_EQ(a2.dimension() - 1, x2.dimension());
  for (typename T::dim_type i = 0; i < x2.dimension(); ++i) {
    EXPECT_EQ(a2.size(i), x2.size(i));
  }
  T v2 = T(x2.size()).zero();
  T x2_select = x2.select(x2.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x2_select.reference_begin(),
           end = x2_select.reference_end(); begin != end; ++begin) {
    cxxblas::trmv(
        a2.size(a2.dimension() - 1), a2[begin.position()].data(),
        v2[begin.position()].data(), a2.stride(a2.dimension() - 2),
        v2.stride(v2.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v2.reference_begin(),
           end = v2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(x2(begin.position()), *begin);
  }

  T a3 = uniformDist(T({3, 9, 7, 7}, {1335, 148, 21, 1}));
  T x3 = linalg.trmv(a3);
  EXPECT_EQ(a3.dimension() - 1, x3.dimension());
  for (typename T::dim_type i = 0; i < x3.dimension(); ++i) {
    EXPECT_EQ(a3.size(i), x3.size(i));
  }
  T v3 = T(x3.size()).zero();
  T x3_select = x3.select(x3.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x3_select.reference_begin(),
           end = x3_select.reference_end(); begin != end; ++begin) {
    cxxblas::trmv(
        a3.size(a3.dimension() - 1), a3[begin.position()].data(),
        v3[begin.position()].data(), a3.stride(a3.dimension() - 2),
        v3.stride(v3.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v3.reference_begin(),
           end = v3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(x3(begin.position()), *begin);
  }
}

TEST(LinalgTest, trmvTest) {
  trmvTest< DoubleLinalg >();
  trmvTest< FloatLinalg >();
  trmvTest< DoubleComplexLinalg >();
  trmvTest< FloatComplexLinalg >();
}

template < typename L >
void trsvTest() {
  typedef typename L::tensor_type T;

  L linalg;

  T a1 = uniformDist(T(7, 7));
  T x1 = linalg.trsv(a1);
  EXPECT_EQ(a1.dimension() - 1, x1.dimension());
  EXPECT_EQ(a1.size(0), x1.size(0));
  T v1 = T(x1.size()).zero();
  cxxblas::trsv(a1.size(1), a1.data(), v1.data(), a1.stride(0), v1.stride(0));
  for (typename T::reference_iterator begin = v1.reference_begin(),
           end = v1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(x1(begin.position()), *begin);
  }

  T a2 = uniformDist(T(3, 9, 7, 7));
  T x2 = linalg.trsv(a2);
  EXPECT_EQ(a2.dimension() - 1, x2.dimension());
  for (typename T::dim_type i = 0; i < x2.dimension(); ++i) {
    EXPECT_EQ(a2.size(i), x2.size(i));
  }
  T v2 = T(x2.size()).zero();
  T x2_select = x2.select(x2.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x2_select.reference_begin(),
           end = x2_select.reference_end(); begin != end; ++begin) {
    cxxblas::trsv(
        a2.size(a2.dimension() - 1), a2[begin.position()].data(),
        v2[begin.position()].data(), a2.stride(a2.dimension() - 2),
        v2.stride(v2.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v2.reference_begin(),
           end = v2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(x2(begin.position()), *begin);
  }

  T a3 = uniformDist(T({3, 9, 7, 7}, {1335, 148, 21, 1}));
  T x3 = linalg.trsv(a3);
  EXPECT_EQ(a3.dimension() - 1, x3.dimension());
  for (typename T::dim_type i = 0; i < x3.dimension(); ++i) {
    EXPECT_EQ(a3.size(i), x3.size(i));
  }
  T v3 = T(x3.size()).zero();
  T x3_select = x3.select(x3.dimension() - 1, 0);
  for (typename T::reference_iterator begin = x3_select.reference_begin(),
           end = x3_select.reference_end(); begin != end; ++begin) {
    cxxblas::trsv(
        a3.size(a3.dimension() - 1), a3[begin.position()].data(),
        v3[begin.position()].data(), a3.stride(a3.dimension() - 2),
        v3.stride(v3.dimension() - 1));
  }
  for (typename T::reference_iterator begin = v3.reference_begin(),
           end = v3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(x3(begin.position()), *begin);
  }
}

TEST(LinalgTest, trsvTest) {
  trsvTest< DoubleLinalg >();
  trsvTest< FloatLinalg >();
  trsvTest< DoubleComplexLinalg >();
  trsvTest< FloatComplexLinalg >();
}

}  // namespace
}  // namespace linalg
}  // namespace thunder
