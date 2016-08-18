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
void gemmTest() {
  typedef typename L::tensor_type T;
  typedef typename L::value_type D;

  L linalg;
  D alpha = D(1.57);
  D beta = D(1.36);

  T a1 = uniformDist(T(7, 10));
  T b1 = uniformDist(T(10, 9));
  T c1 = linalg.gemm(a1, b1, alpha, beta);
  EXPECT_EQ(a1.dimension(), c1.dimension());
  for (typename T::dim_type i = 0; i < a1.dimension() - 2; ++i) {
    EXPECT_EQ(a1.size(i), c1.size(i));
  }
  EXPECT_EQ(a1.size(a1.dimension() - 2), c1.size(c1.dimension() - 2));
  EXPECT_EQ(b1.size(b1.dimension() - 1), c1.size(c1.dimension() - 1));
  T d1 = T(c1.size()).zero();
  T d1_select = d1.select(d1.dimension() - 1, 0).select(d1.dimension() - 2, 0);
  for (typename T::reference_iterator begin = d1_select.reference_begin(),
           end = d1_select.reference_end(); begin != end; ++begin) {
    cxxblas::gemm(
        d1.size(d1.dimension() - 2), d1.size(d1.dimension() - 1),
        a1.size(a1.dimension() - 1), a1[begin.position()].data(),
        b1[begin.position()].data(), d1[begin.position()].data(),
        alpha, beta, a1.stride(a1.dimension() - 2),
        b1.stride(b1.dimension() - 2), d1.stride(d1.dimension() - 2));
  }
  for (typename T::reference_iterator begin = d1.reference_begin(),
           end = d1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(c1(begin.position()), *begin);
  }

  T a2 = uniformDist(T(3, 9, 7, 10));
  T b2 = uniformDist(T(3, 9, 10, 9));
  T c2 = linalg.gemm(a2, b2, alpha, beta);
  EXPECT_EQ(a2.dimension(), c2.dimension());
  for (typename T::dim_type i = 0; i < a2.dimension() - 2; ++i) {
    EXPECT_EQ(a2.size(i), c2.size(i));
  }
  EXPECT_EQ(a2.size(a2.dimension() - 2), c2.size(c2.dimension() - 2));
  EXPECT_EQ(b2.size(b2.dimension() - 1), c2.size(c2.dimension() - 1));
  T d2 = T(c2.size()).zero();
  T d2_select = d2.select(d2.dimension() - 1, 0).select(d2.dimension() - 2, 0);
  for (typename T::reference_iterator begin = d2_select.reference_begin(),
           end = d2_select.reference_end(); begin != end; ++begin) {
    cxxblas::gemm(
        d2.size(d2.dimension() - 2), d2.size(d2.dimension() - 1),
        a2.size(a2.dimension() - 1), a2[begin.position()].data(),
        b2[begin.position()].data(), d2[begin.position()].data(),
        alpha, beta, a2.stride(a2.dimension() - 2),
        b2.stride(b2.dimension() - 2), d2.stride(d2.dimension() - 2));
  }
  for (typename T::reference_iterator begin = d2.reference_begin(),
           end = d2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(c2(begin.position()), *begin);
  }

  T a3 = uniformDist(T({3, 9, 7, 10}, {1335, 148, 21, 1}));
  T b3 = uniformDist(T({3, 9, 10, 9}, {912, 101, 10, 1}));
  T c3 = linalg.gemm(a3, b3, alpha, beta);
  EXPECT_EQ(a3.dimension(), c3.dimension());
  for (typename T::dim_type i = 0; i < a3.dimension() - 2; ++i) {
    EXPECT_EQ(a3.size(i), c3.size(i));
  }
  EXPECT_EQ(a3.size(a3.dimension() - 2), c3.size(c3.dimension() - 2));
  EXPECT_EQ(b3.size(b3.dimension() - 1), c3.size(c3.dimension() - 1));
  T d3 = T(c3.size()).zero();
  T d3_select = d3.select(d3.dimension() - 1, 0).select(d3.dimension() - 2, 0);
  for (typename T::reference_iterator begin = d3_select.reference_begin(),
           end = d3_select.reference_end(); begin != end; ++begin) {
    cxxblas::gemm(
        d3.size(d3.dimension() - 2), d3.size(d3.dimension() - 1),
        a3.size(a3.dimension() - 1), a3[begin.position()].data(),
        b3[begin.position()].data(), d3[begin.position()].data(),
        alpha, beta, a3.stride(a3.dimension() - 2),
        b3.stride(b3.dimension() - 2), d3.stride(d3.dimension() - 2));
  }
  for (typename T::reference_iterator begin = d3.reference_begin(),
           end = d3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(c3(begin.position()), *begin);
  }
}

TEST(LinalgTest, gemmTest) {
  gemmTest< DoubleLinalg >();
  gemmTest< FloatLinalg >();
  gemmTest< DoubleComplexLinalg >();
  gemmTest< FloatComplexLinalg >();
}

template < typename L >
void hemmTest() {
  typedef typename L::tensor_type T;
  typedef typename L::value_type D;

  L linalg;
  D alpha = D(1.57);
  D beta = D(1.36);

  T a1 = uniformDist(T(10, 10));
  T b1 = uniformDist(T(10, 9));
  T c1 = linalg.hemm(a1, b1, alpha, beta);
  EXPECT_EQ(a1.dimension(), c1.dimension());
  for (typename T::dim_type i = 0; i < a1.dimension() - 2; ++i) {
    EXPECT_EQ(a1.size(i), c1.size(i));
  }
  EXPECT_EQ(b1.size(a1.dimension() - 2), c1.size(c1.dimension() - 2));
  EXPECT_EQ(b1.size(b1.dimension() - 1), c1.size(c1.dimension() - 1));
  T d1 = T(c1.size()).zero();
  T d1_select = d1.select(d1.dimension() - 1, 0).select(d1.dimension() - 2, 0);
  for (typename T::reference_iterator begin = d1_select.reference_begin(),
           end = d1_select.reference_end(); begin != end; ++begin) {
    cxxblas::hemm(
        d1.size(d1.dimension() - 2), d1.size(d1.dimension() - 1),
        a1[begin.position()].data(), b1[begin.position()].data(),
        d1[begin.position()].data(), alpha, beta, a1.stride(a1.dimension() - 2),
        b1.stride(b1.dimension() - 2), d1.stride(d1.dimension() - 2));
  }
  for (typename T::reference_iterator begin = d1.reference_begin(),
           end = d1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(c1(begin.position()), *begin);
  }

  T a2 = uniformDist(T(3, 9, 10, 10));
  T b2 = uniformDist(T(3, 9, 10, 9));
  T c2 = linalg.hemm(a2, b2, alpha, beta);
  EXPECT_EQ(a2.dimension(), c2.dimension());
  for (typename T::dim_type i = 0; i < a2.dimension() - 2; ++i) {
    EXPECT_EQ(a2.size(i), c2.size(i));
  }
  EXPECT_EQ(b2.size(b2.dimension() - 2), c2.size(c2.dimension() - 2));
  EXPECT_EQ(b2.size(b2.dimension() - 1), c2.size(c2.dimension() - 1));
  T d2 = T(c2.size()).zero();
  T d2_select = d2.select(d2.dimension() - 1, 0).select(d2.dimension() - 2, 0);
  for (typename T::reference_iterator begin = d2_select.reference_begin(),
           end = d2_select.reference_end(); begin != end; ++begin) {
    cxxblas::hemm(
        d2.size(d2.dimension() - 2), d2.size(d2.dimension() - 1),
        a2[begin.position()].data(), b2[begin.position()].data(),
        d2[begin.position()].data(), alpha, beta, a2.stride(a2.dimension() - 2),
        b2.stride(b2.dimension() - 2), d2.stride(d2.dimension() - 2));
  }
  for (typename T::reference_iterator begin = d2.reference_begin(),
           end = d2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(c2(begin.position()), *begin);
  }

  T a3 = uniformDist(T({3, 9, 10, 10}, {1900, 211, 21, 1}));
  T b3 = uniformDist(T({3, 9, 10, 9}, {912, 101, 10, 1}));
  T c3 = linalg.hemm(a3, b3, alpha, beta);
  EXPECT_EQ(a3.dimension(), c3.dimension());
  for (typename T::dim_type i = 0; i < a3.dimension() - 2; ++i) {
    EXPECT_EQ(a3.size(i), c3.size(i));
  }
  EXPECT_EQ(b3.size(a3.dimension() - 2), c3.size(c3.dimension() - 2));
  EXPECT_EQ(b3.size(b3.dimension() - 1), c3.size(c3.dimension() - 1));
  T d3 = T(c3.size()).zero();
  T d3_select = d3.select(d3.dimension() - 1, 0).select(d3.dimension() - 2, 0);
  for (typename T::reference_iterator begin = d3_select.reference_begin(),
           end = d3_select.reference_end(); begin != end; ++begin) {
    cxxblas::hemm(
        d3.size(d3.dimension() - 2), d3.size(d3.dimension() - 1),
        a3[begin.position()].data(), b3[begin.position()].data(),
        d3[begin.position()].data(), alpha, beta, a3.stride(a3.dimension() - 2),
        b3.stride(b3.dimension() - 2), d3.stride(d3.dimension() - 2));
  }
  for (typename T::reference_iterator begin = d3.reference_begin(),
           end = d3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(c3(begin.position()), *begin);
  }
}

TEST(LinalgTest, hemmTest) {
  hemmTest< DoubleLinalg >();
  hemmTest< FloatLinalg >();
  hemmTest< DoubleComplexLinalg >();
  hemmTest< FloatComplexLinalg >();
}

template < typename L >
void herkTest() {
  typedef typename L::tensor_type T;
  typedef typename L::real_type R;

  L linalg;
  R alpha = R(1.57);
  R beta = R(1.36);

  T a1 = uniformDist(T(10, 7));
  T c1 = linalg.herk(a1, alpha, beta);
  EXPECT_EQ(a1.dimension(), c1.dimension());
  for (typename T::dim_type i = 0; i < a1.dimension() - 2; ++i) {
    EXPECT_EQ(a1.size(i), c1.size(i));
  }
  EXPECT_EQ(a1.size(a1.dimension() - 2), c1.size(c1.dimension() - 2));
  EXPECT_EQ(a1.size(a1.dimension() - 2), c1.size(c1.dimension() - 1));
  T d1 = T(c1.size()).zero();
  T d1_select = d1.select(d1.dimension() - 1, 0).select(d1.dimension() - 2, 0);
  for (typename T::reference_iterator begin = d1_select.reference_begin(),
           end = d1_select.reference_end(); begin != end; ++begin) {
    cxxblas::herk(
        a1.size(a1.dimension() - 2), a1.size(a1.dimension() - 1),
        a1[begin.position()].data(), d1[begin.position()].data(),
        alpha, beta, a1.stride(a1.dimension() - 2),
        d1.stride(d1.dimension() - 2));
  }
  for (typename T::reference_iterator begin = d1.reference_begin(),
           end = d1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(c1(begin.position()), *begin);
  }

  T a2 = uniformDist(T(3, 9, 10, 7));
  T c2 = linalg.herk(a2, alpha, beta);
  EXPECT_EQ(a2.dimension(), c2.dimension());
  for (typename T::dim_type i = 0; i < a2.dimension() - 2; ++i) {
    EXPECT_EQ(a2.size(i), c2.size(i));
  }
  EXPECT_EQ(a2.size(a2.dimension() - 2), c2.size(c2.dimension() - 2));
  EXPECT_EQ(a2.size(a2.dimension() - 2), c2.size(c2.dimension() - 1));
  T d2 = T(c2.size()).zero();
  T d2_select = d2.select(d2.dimension() - 1, 0).select(d2.dimension() - 2, 0);
  for (typename T::reference_iterator begin = d2_select.reference_begin(),
           end = d2_select.reference_end(); begin != end; ++begin) {
    cxxblas::herk(
        a2.size(a2.dimension() - 2), a2.size(a2.dimension() - 1),
        a2[begin.position()].data(), d2[begin.position()].data(), alpha, beta,
        a2.stride(a2.dimension() - 2), d2.stride(d2.dimension() - 2));
  }
  for (typename T::reference_iterator begin = d2.reference_begin(),
           end = d2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(c2(begin.position()), *begin);
  }

  T a3 = uniformDist(T({3, 9, 10, 10}, {1900, 211, 21, 1}));
  T c3 = linalg.herk(a3, alpha, beta);
  EXPECT_EQ(a3.dimension(), c3.dimension());
  for (typename T::dim_type i = 0; i < a3.dimension() - 2; ++i) {
    EXPECT_EQ(a3.size(i), c3.size(i));
  }
  EXPECT_EQ(a3.size(a3.dimension() - 2), c3.size(c3.dimension() - 2));
  EXPECT_EQ(a3.size(a3.dimension() - 2), c3.size(c3.dimension() - 1));
  T d3 = T(c3.size()).zero();
  T d3_select = d3.select(d3.dimension() - 1, 0).select(d3.dimension() - 2, 0);
  for (typename T::reference_iterator begin = d3_select.reference_begin(),
           end = d3_select.reference_end(); begin != end; ++begin) {
    cxxblas::herk(
        a3.size(a3.dimension() - 2), a3.size(a3.dimension() - 1),
        a3[begin.position()].data(), d3[begin.position()].data(), alpha, beta,
        a3.stride(a3.dimension() - 2), d3.stride(d3.dimension() - 2));
  }
  for (typename T::reference_iterator begin = d3.reference_begin(),
           end = d3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(c3(begin.position()), *begin);
  }
}

TEST(LinalgTest, herkTest) {
  herkTest< DoubleLinalg >();
  herkTest< FloatLinalg >();
  herkTest< DoubleComplexLinalg >();
  herkTest< FloatComplexLinalg >();
}

template < typename L >
void her2kTest() {
  typedef typename L::tensor_type T;
  typedef typename L::value_type D;
  typedef typename L::real_type R;

  L linalg;
  D alpha = D(1.57);
  R beta = R(1.36);

  T a1 = uniformDist(T(7, 10));
  T b1 = uniformDist(T(7, 10));
  T c1 = linalg.her2k(a1, b1, alpha, beta);
  EXPECT_EQ(a1.dimension(), c1.dimension());
  for (typename T::dim_type i = 0; i < a1.dimension() - 2; ++i) {
    EXPECT_EQ(a1.size(i), c1.size(i));
  }
  EXPECT_EQ(a1.size(a1.dimension() - 2), c1.size(c1.dimension() - 2));
  EXPECT_EQ(b1.size(b1.dimension() - 2), c1.size(c1.dimension() - 1));
  T d1 = T(c1.size()).zero();
  T d1_select = d1.select(d1.dimension() - 1, 0).select(d1.dimension() - 2, 0);
  for (typename T::reference_iterator begin = d1_select.reference_begin(),
           end = d1_select.reference_end(); begin != end; ++begin) {
    cxxblas::her2k(
        a1.size(a1.dimension() - 2), a1.size(a1.dimension() - 1),
        a1[begin.position()].data(), b1[begin.position()].data(),
        d1[begin.position()].data(), alpha, beta, a1.stride(a1.dimension() - 2),
        b1.stride(b1.dimension() - 2), d1.stride(d1.dimension() - 2));
  }
  for (typename T::reference_iterator begin = d1.reference_begin(),
           end = d1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(c1(begin.position()), *begin);
  }

  T a2 = uniformDist(T(3, 9, 7, 10));
  T b2 = uniformDist(T(3, 9, 7, 10));
  T c2 = linalg.her2k(a2, b2, alpha, beta);
  EXPECT_EQ(a2.dimension(), c2.dimension());
  for (typename T::dim_type i = 0; i < a2.dimension() - 2; ++i) {
    EXPECT_EQ(a2.size(i), c2.size(i));
  }
  EXPECT_EQ(a2.size(a2.dimension() - 2), c2.size(c2.dimension() - 2));
  EXPECT_EQ(b2.size(b2.dimension() - 2), c2.size(c2.dimension() - 1));
  T d2 = T(c2.size()).zero();
  T d2_select = d2.select(d2.dimension() - 1, 0).select(d2.dimension() - 2, 0);
  for (typename T::reference_iterator begin = d2_select.reference_begin(),
           end = d2_select.reference_end(); begin != end; ++begin) {
    cxxblas::her2k(
        a2.size(a2.dimension() - 2), a2.size(a2.dimension() - 1),
        a2[begin.position()].data(), b2[begin.position()].data(),
        d2[begin.position()].data(), alpha, beta, a2.stride(a2.dimension() - 2),
        b2.stride(b2.dimension() - 2), d2.stride(d2.dimension() - 2));
  }
  for (typename T::reference_iterator begin = d2.reference_begin(),
           end = d2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(c2(begin.position()), *begin);
  }

  T a3 = uniformDist(T({3, 9, 7, 10}, {1335, 148, 21, 1}));
  T b3 = uniformDist(T({3, 9, 7, 10}, {912, 101, 10, 1}));
  T c3 = linalg.her2k(a3, b3, alpha, beta);
  EXPECT_EQ(a3.dimension(), c3.dimension());
  for (typename T::dim_type i = 0; i < a3.dimension() - 2; ++i) {
    EXPECT_EQ(a3.size(i), c3.size(i));
  }
  EXPECT_EQ(a3.size(a3.dimension() - 2), c3.size(c3.dimension() - 2));
  EXPECT_EQ(b3.size(b3.dimension() - 2), c3.size(c3.dimension() - 1));
  T d3 = T(c3.size()).zero();
  T d3_select = d3.select(d3.dimension() - 1, 0).select(d3.dimension() - 2, 0);
  for (typename T::reference_iterator begin = d3_select.reference_begin(),
           end = d3_select.reference_end(); begin != end; ++begin) {
    cxxblas::her2k(
        a3.size(a3.dimension() - 2), a3.size(a3.dimension() - 1),
        a3[begin.position()].data(), b3[begin.position()].data(),
        d3[begin.position()].data(), alpha, beta, a3.stride(a3.dimension() - 2),
        b3.stride(b3.dimension() - 2), d3.stride(d3.dimension() - 2));
  }
  for (typename T::reference_iterator begin = d3.reference_begin(),
           end = d3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(c3(begin.position()), *begin);
  }
}

TEST(LinalgTest, her2kTest) {
  her2kTest< DoubleLinalg >();
  her2kTest< FloatLinalg >();
  her2kTest< DoubleComplexLinalg >();
  her2kTest< FloatComplexLinalg >();
}

template < typename L >
void symmTest() {
  typedef typename L::tensor_type T;
  typedef typename L::value_type D;

  L linalg;
  D alpha = D(1.57);
  D beta = D(1.36);

  T a1 = uniformDist(T(10, 10));
  T b1 = uniformDist(T(10, 9));
  T c1 = linalg.symm(a1, b1, alpha, beta);
  EXPECT_EQ(a1.dimension(), c1.dimension());
  for (typename T::dim_type i = 0; i < a1.dimension() - 2; ++i) {
    EXPECT_EQ(a1.size(i), c1.size(i));
  }
  EXPECT_EQ(b1.size(a1.dimension() - 2), c1.size(c1.dimension() - 2));
  EXPECT_EQ(b1.size(b1.dimension() - 1), c1.size(c1.dimension() - 1));
  T d1 = T(c1.size()).zero();
  T d1_select = d1.select(d1.dimension() - 1, 0).select(d1.dimension() - 2, 0);
  for (typename T::reference_iterator begin = d1_select.reference_begin(),
           end = d1_select.reference_end(); begin != end; ++begin) {
    cxxblas::symm(
        d1.size(d1.dimension() - 2), d1.size(d1.dimension() - 1),
        a1[begin.position()].data(), b1[begin.position()].data(),
        d1[begin.position()].data(), alpha, beta, a1.stride(a1.dimension() - 2),
        b1.stride(b1.dimension() - 2), d1.stride(d1.dimension() - 2));
  }
  for (typename T::reference_iterator begin = d1.reference_begin(),
           end = d1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(c1(begin.position()), *begin);
  }

  T a2 = uniformDist(T(3, 9, 10, 10));
  T b2 = uniformDist(T(3, 9, 10, 9));
  T c2 = linalg.symm(a2, b2, alpha, beta);
  EXPECT_EQ(a2.dimension(), c2.dimension());
  for (typename T::dim_type i = 0; i < a2.dimension() - 2; ++i) {
    EXPECT_EQ(a2.size(i), c2.size(i));
  }
  EXPECT_EQ(b2.size(b2.dimension() - 2), c2.size(c2.dimension() - 2));
  EXPECT_EQ(b2.size(b2.dimension() - 1), c2.size(c2.dimension() - 1));
  T d2 = T(c2.size()).zero();
  T d2_select = d2.select(d2.dimension() - 1, 0).select(d2.dimension() - 2, 0);
  for (typename T::reference_iterator begin = d2_select.reference_begin(),
           end = d2_select.reference_end(); begin != end; ++begin) {
    cxxblas::symm(
        d2.size(d2.dimension() - 2), d2.size(d2.dimension() - 1),
        a2[begin.position()].data(), b2[begin.position()].data(),
        d2[begin.position()].data(), alpha, beta, a2.stride(a2.dimension() - 2),
        b2.stride(b2.dimension() - 2), d2.stride(d2.dimension() - 2));
  }
  for (typename T::reference_iterator begin = d2.reference_begin(),
           end = d2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(c2(begin.position()), *begin);
  }

  T a3 = uniformDist(T({3, 9, 10, 10}, {1900, 211, 21, 1}));
  T b3 = uniformDist(T({3, 9, 10, 9}, {912, 101, 10, 1}));
  T c3 = linalg.symm(a3, b3, alpha, beta);
  EXPECT_EQ(a3.dimension(), c3.dimension());
  for (typename T::dim_type i = 0; i < a3.dimension() - 2; ++i) {
    EXPECT_EQ(a3.size(i), c3.size(i));
  }
  EXPECT_EQ(b3.size(a3.dimension() - 2), c3.size(c3.dimension() - 2));
  EXPECT_EQ(b3.size(b3.dimension() - 1), c3.size(c3.dimension() - 1));
  T d3 = T(c3.size()).zero();
  T d3_select = d3.select(d3.dimension() - 1, 0).select(d3.dimension() - 2, 0);
  for (typename T::reference_iterator begin = d3_select.reference_begin(),
           end = d3_select.reference_end(); begin != end; ++begin) {
    cxxblas::symm(
        d3.size(d3.dimension() - 2), d3.size(d3.dimension() - 1),
        a3[begin.position()].data(), b3[begin.position()].data(),
        d3[begin.position()].data(), alpha, beta, a3.stride(a3.dimension() - 2),
        b3.stride(b3.dimension() - 2), d3.stride(d3.dimension() - 2));
  }
  for (typename T::reference_iterator begin = d3.reference_begin(),
           end = d3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(c3(begin.position()), *begin);
  }
}

TEST(LinalgTest, symmTest) {
  symmTest< DoubleLinalg >();
  symmTest< FloatLinalg >();
  symmTest< DoubleComplexLinalg >();
  symmTest< FloatComplexLinalg >();
}

template < typename L >
void syrkTest() {
  typedef typename L::tensor_type T;
  typedef typename L::value_type D;

  L linalg;
  D alpha = D(1.57);
  D beta = D(1.36);

  T a1 = uniformDist(T(10, 7));
  T c1 = linalg.syrk(a1, alpha, beta);
  EXPECT_EQ(a1.dimension(), c1.dimension());
  for (typename T::dim_type i = 0; i < a1.dimension() - 2; ++i) {
    EXPECT_EQ(a1.size(i), c1.size(i));
  }
  EXPECT_EQ(a1.size(a1.dimension() - 2), c1.size(c1.dimension() - 2));
  EXPECT_EQ(a1.size(a1.dimension() - 2), c1.size(c1.dimension() - 1));
  T d1 = T(c1.size()).zero();
  T d1_select = d1.select(d1.dimension() - 1, 0).select(d1.dimension() - 2, 0);
  for (typename T::reference_iterator begin = d1_select.reference_begin(),
           end = d1_select.reference_end(); begin != end; ++begin) {
    cxxblas::syrk(
        a1.size(a1.dimension() - 2), a1.size(a1.dimension() - 1),
        a1[begin.position()].data(), d1[begin.position()].data(),
        alpha, beta, a1.stride(a1.dimension() - 2),
        d1.stride(d1.dimension() - 2));
  }
  for (typename T::reference_iterator begin = d1.reference_begin(),
           end = d1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(c1(begin.position()), *begin);
  }

  T a2 = uniformDist(T(3, 9, 10, 7));
  T c2 = linalg.syrk(a2, alpha, beta);
  EXPECT_EQ(a2.dimension(), c2.dimension());
  for (typename T::dim_type i = 0; i < a2.dimension() - 2; ++i) {
    EXPECT_EQ(a2.size(i), c2.size(i));
  }
  EXPECT_EQ(a2.size(a2.dimension() - 2), c2.size(c2.dimension() - 2));
  EXPECT_EQ(a2.size(a2.dimension() - 2), c2.size(c2.dimension() - 1));
  T d2 = T(c2.size()).zero();
  T d2_select = d2.select(d2.dimension() - 1, 0).select(d2.dimension() - 2, 0);
  for (typename T::reference_iterator begin = d2_select.reference_begin(),
           end = d2_select.reference_end(); begin != end; ++begin) {
    cxxblas::syrk(
        a2.size(a2.dimension() - 2), a2.size(a2.dimension() - 1),
        a2[begin.position()].data(), d2[begin.position()].data(), alpha, beta,
        a2.stride(a2.dimension() - 2), d2.stride(d2.dimension() - 2));
  }
  for (typename T::reference_iterator begin = d2.reference_begin(),
           end = d2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(c2(begin.position()), *begin);
  }

  T a3 = uniformDist(T({3, 9, 10, 10}, {1900, 211, 21, 1}));
  T c3 = linalg.syrk(a3, alpha, beta);
  EXPECT_EQ(a3.dimension(), c3.dimension());
  for (typename T::dim_type i = 0; i < a3.dimension() - 2; ++i) {
    EXPECT_EQ(a3.size(i), c3.size(i));
  }
  EXPECT_EQ(a3.size(a3.dimension() - 2), c3.size(c3.dimension() - 2));
  EXPECT_EQ(a3.size(a3.dimension() - 2), c3.size(c3.dimension() - 1));
  T d3 = T(c3.size()).zero();
  T d3_select = d3.select(d3.dimension() - 1, 0).select(d3.dimension() - 2, 0);
  for (typename T::reference_iterator begin = d3_select.reference_begin(),
           end = d3_select.reference_end(); begin != end; ++begin) {
    cxxblas::syrk(
        a3.size(a3.dimension() - 2), a3.size(a3.dimension() - 1),
        a3[begin.position()].data(), d3[begin.position()].data(), alpha, beta,
        a3.stride(a3.dimension() - 2), d3.stride(d3.dimension() - 2));
  }
  for (typename T::reference_iterator begin = d3.reference_begin(),
           end = d3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(c3(begin.position()), *begin);
  }
}

TEST(LinalgTest, syrkTest) {
  syrkTest< DoubleLinalg >();
  syrkTest< FloatLinalg >();
  syrkTest< DoubleComplexLinalg >();
  syrkTest< FloatComplexLinalg >();
}

template < typename L >
void syr2kTest() {
  typedef typename L::tensor_type T;
  typedef typename L::value_type D;

  L linalg;
  D alpha = D(1.57);
  D beta = D(1.36);

  T a1 = uniformDist(T(7, 10));
  T b1 = uniformDist(T(7, 10));
  T c1 = linalg.syr2k(a1, b1, alpha, beta);
  EXPECT_EQ(a1.dimension(), c1.dimension());
  for (typename T::dim_type i = 0; i < a1.dimension() - 2; ++i) {
    EXPECT_EQ(a1.size(i), c1.size(i));
  }
  EXPECT_EQ(a1.size(a1.dimension() - 2), c1.size(c1.dimension() - 2));
  EXPECT_EQ(b1.size(b1.dimension() - 2), c1.size(c1.dimension() - 1));
  T d1 = T(c1.size()).zero();
  T d1_select = d1.select(d1.dimension() - 1, 0).select(d1.dimension() - 2, 0);
  for (typename T::reference_iterator begin = d1_select.reference_begin(),
           end = d1_select.reference_end(); begin != end; ++begin) {
    cxxblas::syr2k(
        a1.size(a1.dimension() - 2), a1.size(a1.dimension() - 1),
        a1[begin.position()].data(), b1[begin.position()].data(),
        d1[begin.position()].data(), alpha, beta, a1.stride(a1.dimension() - 2),
        b1.stride(b1.dimension() - 2), d1.stride(d1.dimension() - 2));
  }
  for (typename T::reference_iterator begin = d1.reference_begin(),
           end = d1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(c1(begin.position()), *begin);
  }

  T a2 = uniformDist(T(3, 9, 7, 10));
  T b2 = uniformDist(T(3, 9, 7, 10));
  T c2 = linalg.syr2k(a2, b2, alpha, beta);
  EXPECT_EQ(a2.dimension(), c2.dimension());
  for (typename T::dim_type i = 0; i < a2.dimension() - 2; ++i) {
    EXPECT_EQ(a2.size(i), c2.size(i));
  }
  EXPECT_EQ(a2.size(a2.dimension() - 2), c2.size(c2.dimension() - 2));
  EXPECT_EQ(b2.size(b2.dimension() - 2), c2.size(c2.dimension() - 1));
  T d2 = T(c2.size()).zero();
  T d2_select = d2.select(d2.dimension() - 1, 0).select(d2.dimension() - 2, 0);
  for (typename T::reference_iterator begin = d2_select.reference_begin(),
           end = d2_select.reference_end(); begin != end; ++begin) {
    cxxblas::syr2k(
        a2.size(a2.dimension() - 2), a2.size(a2.dimension() - 1),
        a2[begin.position()].data(), b2[begin.position()].data(),
        d2[begin.position()].data(), alpha, beta, a2.stride(a2.dimension() - 2),
        b2.stride(b2.dimension() - 2), d2.stride(d2.dimension() - 2));
  }
  for (typename T::reference_iterator begin = d2.reference_begin(),
           end = d2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(c2(begin.position()), *begin);
  }

  T a3 = uniformDist(T({3, 9, 7, 10}, {1335, 148, 21, 1}));
  T b3 = uniformDist(T({3, 9, 7, 10}, {912, 101, 10, 1}));
  T c3 = linalg.syr2k(a3, b3, alpha, beta);
  EXPECT_EQ(a3.dimension(), c3.dimension());
  for (typename T::dim_type i = 0; i < a3.dimension() - 2; ++i) {
    EXPECT_EQ(a3.size(i), c3.size(i));
  }
  EXPECT_EQ(a3.size(a3.dimension() - 2), c3.size(c3.dimension() - 2));
  EXPECT_EQ(b3.size(b3.dimension() - 2), c3.size(c3.dimension() - 1));
  T d3 = T(c3.size()).zero();
  T d3_select = d3.select(d3.dimension() - 1, 0).select(d3.dimension() - 2, 0);
  for (typename T::reference_iterator begin = d3_select.reference_begin(),
           end = d3_select.reference_end(); begin != end; ++begin) {
    cxxblas::syr2k(
        a3.size(a3.dimension() - 2), a3.size(a3.dimension() - 1),
        a3[begin.position()].data(), b3[begin.position()].data(),
        d3[begin.position()].data(), alpha, beta, a3.stride(a3.dimension() - 2),
        b3.stride(b3.dimension() - 2), d3.stride(d3.dimension() - 2));
  }
  for (typename T::reference_iterator begin = d3.reference_begin(),
           end = d3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(c3(begin.position()), *begin);
  }
}

TEST(LinalgTest, syr2kTest) {
  syr2kTest< DoubleLinalg >();
  syr2kTest< FloatLinalg >();
  syr2kTest< DoubleComplexLinalg >();
  syr2kTest< FloatComplexLinalg >();
}

template < typename L >
void trmmTest() {
  typedef typename L::tensor_type T;
  typedef typename L::value_type D;

  L linalg;
  D alpha = D(1.57);

  T a1 = uniformDist(T(7, 7));
  T b1 = linalg.trmm(a1, alpha);
  EXPECT_EQ(a1.dimension(), b1.dimension());
  for (typename T::dim_type i = 0; i < a1.dimension() - 2; ++i) {
    EXPECT_EQ(a1.size(i), b1.size(i));
  }
  EXPECT_EQ(a1.size(a1.dimension() - 2), b1.size(b1.dimension() - 2));
  EXPECT_EQ(a1.size(a1.dimension() - 1), b1.size(b1.dimension() - 1));
  T d1 = T(b1.size()).zero();
  T d1_select = d1.select(d1.dimension() - 1, 0).select(d1.dimension() - 2, 0);
  for (typename T::reference_iterator begin = d1_select.reference_begin(),
           end = d1_select.reference_end(); begin != end; ++begin) {
    cxxblas::trmm(
        d1.size(d1.dimension() - 2), d1.size(d1.dimension() - 1),
        a1[begin.position()].data(), d1[begin.position()].data(),
        alpha, a1.stride(a1.dimension() - 2), d1.stride(d1.dimension() - 2));
  }
  for (typename T::reference_iterator begin = d1.reference_begin(),
           end = d1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(b1(begin.position()), *begin);
  }

  T a2 = uniformDist(T(3, 9, 7, 7));
  T b2 = linalg.trmm(a2, alpha);
  EXPECT_EQ(a2.dimension(), b2.dimension());
  for (typename T::dim_type i = 0; i < a2.dimension() - 2; ++i) {
    EXPECT_EQ(a2.size(i), b2.size(i));
  }
  EXPECT_EQ(a2.size(a2.dimension() - 2), b2.size(b2.dimension() - 2));
  EXPECT_EQ(a2.size(a2.dimension() - 1), b2.size(b2.dimension() - 1));
  T d2 = T(b2.size()).zero();
  T d2_select = d2.select(d2.dimension() - 1, 0).select(d2.dimension() - 2, 0);
  for (typename T::reference_iterator begin = d2_select.reference_begin(),
           end = d2_select.reference_end(); begin != end; ++begin) {
    cxxblas::trmm(
        d2.size(d2.dimension() - 2), d2.size(d2.dimension() - 1),
        a2[begin.position()].data(), d2[begin.position()].data(),
        alpha, a2.stride(a2.dimension() - 2), d2.stride(d2.dimension() - 2));
  }
  for (typename T::reference_iterator begin = d2.reference_begin(),
           end = d2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(b2(begin.position()), *begin);
  }

  T a3 = uniformDist(T({3, 9, 7, 7}, {1335, 148, 21, 1}));
  T b3 = linalg.trmm(a3, alpha);
  EXPECT_EQ(a3.dimension(), b3.dimension());
  for (typename T::dim_type i = 0; i < a3.dimension() - 2; ++i) {
    EXPECT_EQ(a3.size(i), b3.size(i));
  }
  EXPECT_EQ(a3.size(a3.dimension() - 2), b3.size(b3.dimension() - 2));
  EXPECT_EQ(a3.size(a3.dimension() - 1), b3.size(b3.dimension() - 1));
  T d3 = T(b3.size()).zero();
  T d3_select = d3.select(d3.dimension() - 1, 0).select(d3.dimension() - 2, 0);
  for (typename T::reference_iterator begin = d3_select.reference_begin(),
           end = d3_select.reference_end(); begin != end; ++begin) {
    cxxblas::trmm(
        d3.size(d3.dimension() - 2), d3.size(d3.dimension() - 1),
        a3[begin.position()].data(), d3[begin.position()].data(),
        alpha, a3.stride(a3.dimension() - 2), d3.stride(d3.dimension() - 2));
  }
  for (typename T::reference_iterator begin = d3.reference_begin(),
           end = d3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(b3(begin.position()), *begin);
  }
}

TEST(LinalgTest, trmmTest) {
  trmmTest< DoubleLinalg >();
  trmmTest< FloatLinalg >();
  trmmTest< DoubleComplexLinalg >();
  trmmTest< FloatComplexLinalg >();
}

template < typename L >
void trsmTest() {
  typedef typename L::tensor_type T;
  typedef typename L::value_type D;

  L linalg;
  D alpha = D(1.57);

  T a1 = uniformDist(T(7, 7));
  T b1 = linalg.trsm(a1, alpha);
  EXPECT_EQ(a1.dimension(), b1.dimension());
  for (typename T::dim_type i = 0; i < a1.dimension() - 2; ++i) {
    EXPECT_EQ(a1.size(i), b1.size(i));
  }
  EXPECT_EQ(a1.size(a1.dimension() - 2), b1.size(b1.dimension() - 2));
  EXPECT_EQ(a1.size(a1.dimension() - 1), b1.size(b1.dimension() - 1));
  T d1 = T(b1.size()).zero();
  T d1_select = d1.select(d1.dimension() - 1, 0).select(d1.dimension() - 2, 0);
  for (typename T::reference_iterator begin = d1_select.reference_begin(),
           end = d1_select.reference_end(); begin != end; ++begin) {
    cxxblas::trsm(
        d1.size(d1.dimension() - 2), d1.size(d1.dimension() - 1),
        a1[begin.position()].data(), d1[begin.position()].data(),
        alpha, a1.stride(a1.dimension() - 2), d1.stride(d1.dimension() - 2));
  }
  for (typename T::reference_iterator begin = d1.reference_begin(),
           end = d1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(b1(begin.position()), *begin);
  }

  T a2 = uniformDist(T(3, 9, 7, 7));
  T b2 = linalg.trsm(a2, alpha);
  EXPECT_EQ(a2.dimension(), b2.dimension());
  for (typename T::dim_type i = 0; i < a2.dimension() - 2; ++i) {
    EXPECT_EQ(a2.size(i), b2.size(i));
  }
  EXPECT_EQ(a2.size(a2.dimension() - 2), b2.size(b2.dimension() - 2));
  EXPECT_EQ(a2.size(a2.dimension() - 1), b2.size(b2.dimension() - 1));
  T d2 = T(b2.size()).zero();
  T d2_select = d2.select(d2.dimension() - 1, 0).select(d2.dimension() - 2, 0);
  for (typename T::reference_iterator begin = d2_select.reference_begin(),
           end = d2_select.reference_end(); begin != end; ++begin) {
    cxxblas::trsm(
        d2.size(d2.dimension() - 2), d2.size(d2.dimension() - 1),
        a2[begin.position()].data(), d2[begin.position()].data(),
        alpha, a2.stride(a2.dimension() - 2), d2.stride(d2.dimension() - 2));
  }
  for (typename T::reference_iterator begin = d2.reference_begin(),
           end = d2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(b2(begin.position()), *begin);
  }

  T a3 = uniformDist(T({3, 9, 7, 7}, {1335, 148, 21, 1}));
  T b3 = linalg.trsm(a3, alpha);
  EXPECT_EQ(a3.dimension(), b3.dimension());
  for (typename T::dim_type i = 0; i < a3.dimension() - 2; ++i) {
    EXPECT_EQ(a3.size(i), b3.size(i));
  }
  EXPECT_EQ(a3.size(a3.dimension() - 2), b3.size(b3.dimension() - 2));
  EXPECT_EQ(a3.size(a3.dimension() - 1), b3.size(b3.dimension() - 1));
  T d3 = T(b3.size()).zero();
  T d3_select = d3.select(d3.dimension() - 1, 0).select(d3.dimension() - 2, 0);
  for (typename T::reference_iterator begin = d3_select.reference_begin(),
           end = d3_select.reference_end(); begin != end; ++begin) {
    cxxblas::trsm(
        d3.size(d3.dimension() - 2), d3.size(d3.dimension() - 1),
        a3[begin.position()].data(), d3[begin.position()].data(),
        alpha, a3.stride(a3.dimension() - 2), d3.stride(d3.dimension() - 2));
  }
  for (typename T::reference_iterator begin = d3.reference_begin(),
           end = d3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(b3(begin.position()), *begin);
  }
}

TEST(LinalgTest, trsmTest) {
  trsmTest< DoubleLinalg >();
  trsmTest< FloatLinalg >();
  trsmTest< DoubleComplexLinalg >();
  trsmTest< FloatComplexLinalg >();
}

}  // namespace
}  // namespace linalg
}  // namespace thunder
