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
#include "thunder/linalg/cxxblas.hpp"
#include "thunder/tensor.hpp"

namespace thunder {
namespace linalg {
namespace {

template < typename D >
void expectEq(const D &a, const D &b) {
  EXPECT_FLOAT_EQ(a, b);
}
template < typename D >
void expectEq(const ::std::complex< D > &a, const ::std::complex< D > &b) {
  EXPECT_FLOAT_EQ(::std::real(a), ::std::real(b));
  EXPECT_FLOAT_EQ(::std::imag(a), ::std::imag(b));
}

template < typename D >
D conjg(const D &v) {
  return v;
}
template < typename D >
::std::complex< D > conjg(const ::std::complex< D > &v) {
  return ::std::conj(v);
}

template < typename D >
class RandomGenerator {
 public:
  template < typename G, typename R >
  D operator()(G *gen, R *dist) {
    return static_cast< D >((*dist)(*gen));
  }
};

template < typename D >
class RandomGenerator< ::std::complex< D > > {
 public:
  template < typename G, typename R >
  ::std::complex< D > operator()(G *gen, R *dist) {
    return ::std::complex< D >(
        static_cast< D >((*dist)(*gen)), static_cast< D >((*dist)(*gen)));
  }
};

template < typename T >
void asumTest() {
  typedef typename T::value_type D;
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  Linalg< T > linalg;

  T t1(7);
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r1 = linalg.asum(t1);
  D s1 = cxxblas::asum(t1.size(0), t1.data(), t1.stride(0));
  EXPECT_EQ(1, r1.dimension());
  EXPECT_EQ(1, r1.size(0));
  expectEq(s1, r1(0));

  T t2(typename T::size_storage({7}), typename T::stride_storage({2}));
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r2 = linalg.asum(t2);
  D s2 = cxxblas::asum(t2.size(0), t2.data(), t2.stride(0));
  EXPECT_EQ(1, r2.dimension());
  EXPECT_EQ(1, r2.size(0));
  expectEq(s2, r2(0));

  T t3(20, 7);
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r3 = linalg.asum(t3);
  EXPECT_EQ(1, r3.dimension());
  EXPECT_EQ(20, r3.size(0));
  for (int i = 0; i < 20; ++i) {
    D s3 = cxxblas::asum(t3[i].size(0), t3[i].data(), t3[i].stride(0));
    expectEq(s3, r3(i));
  }

  T t4({20, 7}, {14, 2});
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r4 = linalg.asum(t4);
  EXPECT_EQ(1, r4.dimension());
  EXPECT_EQ(20, r4.size(0));
  for (int i = 0; i < 20; ++i) {
    D s4 = cxxblas::asum(t4[i].size(0), t4[i].data(), t4[i].stride(0));
    expectEq(s4, r4(i));
  }

  T t5(10, 20, 7);
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r5 = linalg.asum(t5);
  EXPECT_EQ(2, r5.dimension());
  EXPECT_EQ(10, r5.size(0));
  EXPECT_EQ(20, r5.size(1));
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 20; ++j) {
      D s5 = cxxblas::asum(
          t5[i][j].size(0), t5[i][j].data(), t5[i][j].stride(0));
      expectEq(s5, r5(i, j));
    }
  }

  T t6({10, 20, 7}, {302, 15, 2});
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r6 = linalg.asum(t6);
  EXPECT_EQ(2, r6.dimension());
  EXPECT_EQ(10, r6.size(0));
  EXPECT_EQ(20, r6.size(1));
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 20; ++j) {
      D s6 = cxxblas::asum(
          t6[i][j].size(0), t6[i][j].data(), t6[i][j].stride(0));
      expectEq(s6, r6(i, j));
    }
  }
}

TEST(LinalgTest, asumTest) {
  asumTest< DoubleTensor >();
  asumTest< FloatTensor >();
  asumTest< DoubleComplexTensor >();
  asumTest< FloatComplexTensor >();
}

template < typename T >
void axpyTest() {
  typedef typename T::value_type D;
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  Linalg< T > linalg;

  T t1(7);
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  D a1 = rand(&gen, &dist);
  T r1 = linalg.axpy(t1, a1);
  EXPECT_EQ(t1.dimension(), r1.dimension());
  for (typename T::dim_type i = 0; i < t1.dimension(); ++i) {
    EXPECT_EQ(t1.size(i), r1.size(i));
  }
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    expectEq(a1 * (*begin), r1(begin.position()));
  }
  r1.resize(7);
  for (typename T::reference_iterator begin = r1.reference_begin(),
           end = r1.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T s1 = r1.clone();
  r1 = linalg.axpy(t1, r1, a1);
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    expectEq(a1 * (*begin) + s1(begin.position()), r1(begin.position()));
  }

  T t2(typename T::size_storage({7}), typename T::stride_storage({2}));
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  D a2 = rand(&gen, &dist);
  T r2 = linalg.axpy(t2, a2);
  EXPECT_EQ(t2.dimension(), r2.dimension());
  for (typename T::dim_type i = 0; i < t2.dimension(); ++i) {
    EXPECT_EQ(t2.size(i), r2.size(i));
  }
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    expectEq(a2 * (*begin), r2(begin.position()));
  }
  r2.resize(7);
  for (typename T::reference_iterator begin = r2.reference_begin(),
           end = r2.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T s2 = r2.clone();
  r2 = linalg.axpy(t2, r2, a2);
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    expectEq(a2 * (*begin) + s2(begin.position()), r2(begin.position()));
  }

  T t3(20, 7);
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  D a3 = rand(&gen, &dist);
  T r3 = linalg.axpy(t3, a3);
  EXPECT_EQ(t3.dimension(), r3.dimension());
  for (typename T::dim_type i = 0; i < t3.dimension(); ++i) {
    EXPECT_EQ(t3.size(i), r3.size(i));
  }
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    expectEq(a3 * (*begin), r3(begin.position()));
  }
  r3.resize(20, 7);
  for (typename T::reference_iterator begin = r3.reference_begin(),
           end = r3.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T s3 = r3.clone();
  r3 = linalg.axpy(t3, r3, a3);
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    expectEq(a3 * (*begin) + s3(begin.position()), r3(begin.position()));
  }

  T t4({20, 7}, {15, 2});
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  D a4 = rand(&gen, &dist);
  T r4 = linalg.axpy(t4, a4);
  EXPECT_EQ(t4.dimension(), r4.dimension());
  for (typename T::dim_type i = 0; i < t4.dimension(); ++i) {
    EXPECT_EQ(t4.size(i), r4.size(i));
  }
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    expectEq(a4 * (*begin), r4(begin.position()));
  }
  r4.resize({20, 7}, {14, 1});
  for (typename T::reference_iterator begin = r4.reference_begin(),
           end = r4.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T s4 = r4.clone();
  r4 = linalg.axpy(t4, r4, a4);
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    expectEq(a4 * (*begin) + s4(begin.position()), r4(begin.position()));
  }

  T t5(10, 20, 7);
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  D a5 = rand(&gen, &dist);
  T r5 = linalg.axpy(t5, a5);
  EXPECT_EQ(t5.dimension(), r5.dimension());
  for (typename T::dim_type i = 0; i < t5.dimension(); ++i) {
    EXPECT_EQ(t5.size(i), r5.size(i));
  }
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    expectEq(a5 * (*begin), r5(begin.position()));
  }
  r5.resize(10, 20, 7);
  for (typename T::reference_iterator begin = r5.reference_begin(),
           end = r5.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T s5 = r5.clone();
  r5 = linalg.axpy(t5, r5, a5);
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    expectEq(a5 * (*begin) + s5(begin.position()), r5(begin.position()));
  }

  T t6(10, 20, 7);
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  D a6 = rand(&gen, &dist);
  T r6 = linalg.axpy(t6, a6);
  EXPECT_EQ(t6.dimension(), r6.dimension());
  for (typename T::dim_type i = 0; i < t6.dimension(); ++i) {
    EXPECT_EQ(t6.size(i), r6.size(i));
  }
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    expectEq(a6 * (*begin), r6(begin.position()));
  }
  r6.resize({10, 20, 7}, {309, 15, 2});
  for (typename T::reference_iterator begin = r6.reference_begin(),
           end = r6.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T s6 = r6.clone();
  r6 = linalg.axpy(t6, r6, a6);
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    expectEq(a6 * (*begin) + s6(begin.position()), r6(begin.position()));
  }
}

TEST(LinalgTest, axpyTest) {
  axpyTest< DoubleTensor >();
  axpyTest< FloatTensor >();
  axpyTest< DoubleComplexTensor >();
  axpyTest< FloatComplexTensor >();
}

template < typename T >
void copyTest() {
  typedef typename T::value_type D;
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  Linalg< T > linalg;

  T t1(7);
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r1 = linalg.copy(t1);
  EXPECT_EQ(t1.dimension(), r1.dimension());
  for (typename T::dim_type i = 0; i < t1.dimension(); ++i) {
    EXPECT_EQ(t1.size(i), r1.size(i));
  }
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    expectEq(*begin, r1(begin.position()));
  }

  T t2(typename T::size_storage({7}), typename T::stride_storage({2}));
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r2 = linalg.copy(t2);
  EXPECT_EQ(t2.dimension(), r2.dimension());
  for (typename T::dim_type i = 0; i < t2.dimension(); ++i) {
    EXPECT_EQ(t2.size(i), r2.size(i));
  }
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    expectEq(*begin, r2(begin.position()));
  }

  T t3(20, 7);
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r3 = linalg.copy(t3);
  EXPECT_EQ(t3.dimension(), r3.dimension());
  for (typename T::dim_type i = 0; i < t3.dimension(); ++i) {
    EXPECT_EQ(t3.size(i), r3.size(i));
  }
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    expectEq(*begin, r3(begin.position()));
  }

  T t4({20, 7}, {15, 2});
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r4 = linalg.copy(t4);
  EXPECT_EQ(t4.dimension(), r4.dimension());
  for (typename T::dim_type i = 0; i < t4.dimension(); ++i) {
    EXPECT_EQ(t4.size(i), r4.size(i));
  }
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    expectEq(*begin, r4(begin.position()));
  }

  T t5(10, 20, 7);
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r5 = linalg.copy(t5);
  EXPECT_EQ(t5.dimension(), r5.dimension());
  for (typename T::dim_type i = 0; i < t5.dimension(); ++i) {
    EXPECT_EQ(t5.size(i), r5.size(i));
  }
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    expectEq(*begin, r5(begin.position()));
  }

  T t6({10, 20, 7}, {309, 15, 2});
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r6 = linalg.copy(t6);
  EXPECT_EQ(t6.dimension(), r6.dimension());
  for (typename T::dim_type i = 0; i < t6.dimension(); ++i) {
    EXPECT_EQ(t6.size(i), r6.size(i));
  }
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    expectEq(*begin, r6(begin.position()));
  }
}

TEST(LinalgTest, copyTest) {
  copyTest< DoubleTensor >();
  copyTest< FloatTensor >();
  copyTest< DoubleComplexTensor >();
  copyTest< FloatComplexTensor >();
}

template < typename T >
void dotTest() {
  typedef typename T::value_type D;
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  Linalg< T > linalg;

  T t1(7);
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T s1(7);
  for (typename T::reference_iterator begin = s1.reference_begin(),
           end = s1.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r1 = linalg.dot(t1, s1);
  EXPECT_EQ(1, r1.dimension());
  EXPECT_EQ(1, r1.size(0));
  D u1 = cxxblas::dot(
      t1.size(0), t1.data(), s1.data(), t1.stride(0), s1.stride(0));
  expectEq(u1, r1(0));


  T t2(typename T::size_storage({7}), typename T::stride_storage({2}));
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T s2(typename T::size_storage({7}), typename T::stride_storage({2}));
  for (typename T::reference_iterator begin = s2.reference_begin(),
           end = s2.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r2 = linalg.dot(t2, s2);
  EXPECT_EQ(1, r2.dimension());
  EXPECT_EQ(1, r2.size(0));
  D u2 = cxxblas::dot(
      t2.size(0), t2.data(), s2.data(), t2.stride(0), s2.stride(0));
  expectEq(u2, r2(0));

  T t3(20, 7);
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T s3(20, 7);
  for (typename T::reference_iterator begin = s3.reference_begin(),
           end = s3.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r3 = linalg.dot(t3, s3);
  EXPECT_EQ(t3.dimension() - 1, r3.dimension());
  for (typename T::dim_type i = 0; i < t3.dimension() - 1; ++i) {
    EXPECT_EQ(t3.size(i), r3.size(i));
  }
  for (typename T::reference_iterator begin = r3.reference_begin(),
           end = r3.reference_end(); begin != end; ++begin) {
    D u3 = cxxblas::dot(
        t3[begin.position()].size(0), t3[begin.position()].data(),
        s3[begin.position()].data(), t3[begin.position()].stride(0),
        s3[begin.position()].stride(0));
    expectEq(u3, r3(begin.position()));
  }

  T t4({20, 7}, {15, 2});
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T s4({20, 7}, {25, 3});
  for (typename T::reference_iterator begin = s4.reference_begin(),
           end = s4.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r4 = linalg.dot(t4, s4);
  EXPECT_EQ(t4.dimension() - 1, r4.dimension());
  for (typename T::dim_type i = 0; i < t4.dimension() - 1; ++i) {
    EXPECT_EQ(t4.size(i), r4.size(i));
  }
  for (typename T::reference_iterator begin = r4.reference_begin(),
           end = r4.reference_end(); begin != end; ++begin) {
    D u4 = cxxblas::dot(
        t4[begin.position()].size(0), t4[begin.position()].data(),
        s4[begin.position()].data(), t4[begin.position()].stride(0),
        s4[begin.position()].stride(0));
    expectEq(u4, r4(begin.position()));
  }

  T t5(10, 20, 7);
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T s5(10, 20, 7);
  for (typename T::reference_iterator begin = s5.reference_begin(),
           end = s5.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r5 = linalg.dot(t5, s5);
  EXPECT_EQ(t5.dimension() - 1, r5.dimension());
  for (typename T::dim_type i = 0; i < t5.dimension() - 1; ++i) {
    EXPECT_EQ(t5.size(i), r5.size(i));
  }
  for (typename T::reference_iterator begin = r5.reference_begin(),
           end = r5.reference_end(); begin != end; ++begin) {
    D u5 = cxxblas::dot(
        t5[begin.position()].size(0), t5[begin.position()].data(),
        s5[begin.position()].data(), t5[begin.position()].stride(0),
        s5[begin.position()].stride(0));
    expectEq(u5, r5(begin.position()));
  }

  T t6({10, 20, 7}, {309, 15, 2});
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T s6({10, 20, 7}, {171, 8, 1});
  for (typename T::reference_iterator begin = s6.reference_begin(),
           end = s6.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r6 = linalg.dot(t6, s6);
  EXPECT_EQ(t6.dimension() - 1, r6.dimension());
  for (typename T::dim_type i = 0; i < t6.dimension() - 1; ++i) {
    EXPECT_EQ(t6.size(i), r6.size(i));
  }
  for (typename T::reference_iterator begin = r6.reference_begin(),
           end = r6.reference_end(); begin != end; ++begin) {
    D u6 = cxxblas::dot(
        t6[begin.position()].size(0), t6[begin.position()].data(),
        s6[begin.position()].data(), t6[begin.position()].stride(0),
        s6[begin.position()].stride(0));
    expectEq(u6, r6(begin.position()));
  }

}

TEST(LinalgTest, dotTest) {
  dotTest< DoubleTensor >();
  dotTest< FloatTensor >();
  dotTest< DoubleComplexTensor >();
  dotTest< FloatComplexTensor >();
}

template < typename T >
void dotcTest() {
  typedef typename T::value_type D;
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  Linalg< T > linalg;

  T t1(7);
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T s1(7);
  for (typename T::reference_iterator begin = s1.reference_begin(),
           end = s1.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r1 = linalg.dotc(t1, s1);
  EXPECT_EQ(1, r1.dimension());
  EXPECT_EQ(1, r1.size(0));
  D u1 = cxxblas::dotc(
      t1.size(0), t1.data(), s1.data(), t1.stride(0), s1.stride(0));
  expectEq(u1, r1(0));


  T t2(typename T::size_storage({7}), typename T::stride_storage({2}));
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T s2(typename T::size_storage({7}), typename T::stride_storage({2}));
  for (typename T::reference_iterator begin = s2.reference_begin(),
           end = s2.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r2 = linalg.dotc(t2, s2);
  EXPECT_EQ(1, r2.dimension());
  EXPECT_EQ(1, r2.size(0));
  D u2 = cxxblas::dotc(
      t2.size(0), t2.data(), s2.data(), t2.stride(0), s2.stride(0));
  expectEq(u2, r2(0));

  T t3(20, 7);
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T s3(20, 7);
  for (typename T::reference_iterator begin = s3.reference_begin(),
           end = s3.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r3 = linalg.dotc(t3, s3);
  EXPECT_EQ(t3.dimension() - 1, r3.dimension());
  for (typename T::dim_type i = 0; i < t3.dimension() - 1; ++i) {
    EXPECT_EQ(t3.size(i), r3.size(i));
  }
  for (typename T::reference_iterator begin = r3.reference_begin(),
           end = r3.reference_end(); begin != end; ++begin) {
    D u3 = cxxblas::dotc(
        t3[begin.position()].size(0), t3[begin.position()].data(),
        s3[begin.position()].data(), t3[begin.position()].stride(0),
        s3[begin.position()].stride(0));
    expectEq(u3, r3(begin.position()));
  }

  T t4({20, 7}, {15, 2});
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T s4({20, 7}, {25, 3});
  for (typename T::reference_iterator begin = s4.reference_begin(),
           end = s4.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r4 = linalg.dotc(t4, s4);
  EXPECT_EQ(t4.dimension() - 1, r4.dimension());
  for (typename T::dim_type i = 0; i < t4.dimension() - 1; ++i) {
    EXPECT_EQ(t4.size(i), r4.size(i));
  }
  for (typename T::reference_iterator begin = r4.reference_begin(),
           end = r4.reference_end(); begin != end; ++begin) {
    D u4 = cxxblas::dotc(
        t4[begin.position()].size(0), t4[begin.position()].data(),
        s4[begin.position()].data(), t4[begin.position()].stride(0),
        s4[begin.position()].stride(0));
    expectEq(u4, r4(begin.position()));
  }

  T t5(10, 20, 7);
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T s5(10, 20, 7);
  for (typename T::reference_iterator begin = s5.reference_begin(),
           end = s5.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r5 = linalg.dotc(t5, s5);
  EXPECT_EQ(t5.dimension() - 1, r5.dimension());
  for (typename T::dim_type i = 0; i < t5.dimension() - 1; ++i) {
    EXPECT_EQ(t5.size(i), r5.size(i));
  }
  for (typename T::reference_iterator begin = r5.reference_begin(),
           end = r5.reference_end(); begin != end; ++begin) {
    D u5 = cxxblas::dotc(
        t5[begin.position()].size(0), t5[begin.position()].data(),
        s5[begin.position()].data(), t5[begin.position()].stride(0),
        s5[begin.position()].stride(0));
    expectEq(u5, r5(begin.position()));
  }

  T t6({10, 20, 7}, {309, 15, 2});
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T s6({10, 20, 7}, {171, 8, 1});
  for (typename T::reference_iterator begin = s6.reference_begin(),
           end = s6.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r6 = linalg.dotc(t6, s6);
  EXPECT_EQ(t6.dimension() - 1, r6.dimension());
  for (typename T::dim_type i = 0; i < t6.dimension() - 1; ++i) {
    EXPECT_EQ(t6.size(i), r6.size(i));
  }
  for (typename T::reference_iterator begin = r6.reference_begin(),
           end = r6.reference_end(); begin != end; ++begin) {
    D u6 = cxxblas::dotc(
        t6[begin.position()].size(0), t6[begin.position()].data(),
        s6[begin.position()].data(), t6[begin.position()].stride(0),
        s6[begin.position()].stride(0));
    expectEq(u6, r6(begin.position()));
  }

}

TEST(LinalgTest, dotcTest) {
  dotcTest< DoubleTensor >();
  dotcTest< FloatTensor >();
  dotcTest< DoubleComplexTensor >();
  dotcTest< FloatComplexTensor >();
}

template < typename T >
void nrm2Test() {
  typedef typename T::value_type D;
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  Linalg< T > linalg;

  T t1(7);
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r1 = linalg.nrm2(t1);
  D s1 = cxxblas::nrm2(t1.size(0), t1.data(), t1.stride(0));
  EXPECT_EQ(1, r1.dimension());
  EXPECT_EQ(1, r1.size(0));
  expectEq(s1, r1(0));

  T t2(typename T::size_storage({7}), typename T::stride_storage({2}));
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r2 = linalg.nrm2(t2);
  D s2 = cxxblas::nrm2(t2.size(0), t2.data(), t2.stride(0));
  EXPECT_EQ(1, r2.dimension());
  EXPECT_EQ(1, r2.size(0));
  expectEq(s2, r2(0));

  T t3(20, 7);
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r3 = linalg.nrm2(t3);
  EXPECT_EQ(1, r3.dimension());
  EXPECT_EQ(20, r3.size(0));
  for (int i = 0; i < 20; ++i) {
    D s3 = cxxblas::nrm2(t3[i].size(0), t3[i].data(), t3[i].stride(0));
    expectEq(s3, r3(i));
  }

  T t4({20, 7}, {14, 2});
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r4 = linalg.nrm2(t4);
  EXPECT_EQ(1, r4.dimension());
  EXPECT_EQ(20, r4.size(0));
  for (int i = 0; i < 20; ++i) {
    D s4 = cxxblas::nrm2(t4[i].size(0), t4[i].data(), t4[i].stride(0));
    expectEq(s4, r4(i));
  }

  T t5(10, 20, 7);
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r5 = linalg.nrm2(t5);
  EXPECT_EQ(2, r5.dimension());
  EXPECT_EQ(10, r5.size(0));
  EXPECT_EQ(20, r5.size(1));
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 20; ++j) {
      D s5 = cxxblas::nrm2(
          t5[i][j].size(0), t5[i][j].data(), t5[i][j].stride(0));
      expectEq(s5, r5(i, j));
    }
  }

  T t6({10, 20, 7}, {302, 15, 2});
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T r6 = linalg.nrm2(t6);
  EXPECT_EQ(2, r6.dimension());
  EXPECT_EQ(10, r6.size(0));
  EXPECT_EQ(20, r6.size(1));
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 20; ++j) {
      D s6 = cxxblas::nrm2(
          t6[i][j].size(0), t6[i][j].data(), t6[i][j].stride(0));
      expectEq(s6, r6(i, j));
    }
  }
}

TEST(LinalgTest, nrm2Test) {
  nrm2Test< DoubleTensor >();
  nrm2Test< FloatTensor >();
  nrm2Test< DoubleComplexTensor >();
  nrm2Test< FloatComplexTensor >();
}

template < typename T >
void rotTest() {
  typedef typename T::value_type D;
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  Linalg< T > linalg;

  T t1(7);
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  D c1 = rand(&gen, &dist);
  D s1 = rand(&gen, &dist);
  T u1 = t1.clone();
  T r1 = linalg.rot(t1, c1, s1);
  EXPECT_EQ(t1.dimension(), r1.dimension());
  for (typename T::dim_type i = 0; i < t1.dimension(); ++i) {
    EXPECT_EQ(t1.size(i), r1.size(i));
  }
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    expectEq(::std::real(c1) * u1(begin.position()), *begin);
    expectEq(-::std::real(s1) * u1(begin.position()), r1(begin.position()));
  }
  t1.copy(u1);
  r1.resize(7);
  for (typename T::reference_iterator begin = r1.reference_begin(),
           end = r1.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T v1 = r1.clone();
  r1 = linalg.rot(t1, r1, c1, s1);
  EXPECT_EQ(t1.dimension(), r1.dimension());
  for (typename T::dim_type i = 0; i < t1.dimension(); ++i) {
    EXPECT_EQ(t1.size(i), r1.size(i));
  }
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    D x1 = ::std::real(c1) * u1(begin.position()) +
        ::std::real(s1) * v1(begin.position());
    D y1 = ::std::real(c1) * v1(begin.position()) -
        ::std::real(s1) * u1(begin.position());
    expectEq(x1, *begin);
    expectEq(y1, r1(begin.position()));
  }

  T t2(typename T::size_storage({7}), typename T::stride_storage({2}));
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  D c2 = rand(&gen, &dist);
  D s2 = rand(&gen, &dist);
  T u2 = t2.clone();
  T r2 = linalg.rot(t2, c2, s2);
  EXPECT_EQ(t2.dimension(), r2.dimension());
  for (typename T::dim_type i = 0; i < t2.dimension(); ++i) {
    EXPECT_EQ(t2.size(i), r2.size(i));
  }
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    expectEq(::std::real(c2) * u2(begin.position()), *begin);
    expectEq(-::std::real(s2) * u2(begin.position()), r2(begin.position()));
  }
  t2.copy(u2);
  r2.resize(typename T::size_storage({7}), typename T::stride_storage({3}));
  for (typename T::reference_iterator begin = r2.reference_begin(),
           end = r2.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T v2 = r2.clone();
  r2 = linalg.rot(t2, r2, c2, s2);
  EXPECT_EQ(t2.dimension(), r2.dimension());
  for (typename T::dim_type i = 0; i < t2.dimension(); ++i) {
    EXPECT_EQ(t2.size(i), r2.size(i));
  }
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    D x2 = ::std::real(c2) * u2(begin.position()) +
        ::std::real(s2) * v2(begin.position());
    D y2 = ::std::real(c2) * v2(begin.position()) -
        ::std::real(s2) * u2(begin.position());
    expectEq(x2, *begin);
    expectEq(y2, r2(begin.position()));
  }

  T t3(20, 7);
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  D c3 = rand(&gen, &dist);
  D s3 = rand(&gen, &dist);
  T u3 = t3.clone();
  T r3 = linalg.rot(t3, c3, s3);
  EXPECT_EQ(t3.dimension(), r3.dimension());
  for (typename T::dim_type i = 0; i < t3.dimension(); ++i) {
    EXPECT_EQ(t3.size(i), r3.size(i));
  }
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    expectEq(::std::real(c3) * u3(begin.position()), *begin);
    expectEq(-::std::real(s3) * u3(begin.position()), r3(begin.position()));
  }
  t3.copy(u3);
  r3.resize(20, 7);
  for (typename T::reference_iterator begin = r3.reference_begin(),
           end = r3.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T v3 = r3.clone();
  r3 = linalg.rot(t3, r3, c3, s3);
  EXPECT_EQ(t3.dimension(), r3.dimension());
  for (typename T::dim_type i = 0; i < t3.dimension(); ++i) {
    EXPECT_EQ(t3.size(i), r3.size(i));
  }
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    D x3 = ::std::real(c3) * u3(begin.position()) +
        ::std::real(s3) * v3(begin.position());
    D y3 = ::std::real(c3) * v3(begin.position()) -
        ::std::real(s3) * u3(begin.position());
    expectEq(x3, *begin);
    expectEq(y3, r3(begin.position()));
  }

  T t4({20, 7}, {15, 2});
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  D c4 = rand(&gen, &dist);
  D s4 = rand(&gen, &dist);
  T u4 = t4.clone();
  T r4 = linalg.rot(t4, c4, s4);
  EXPECT_EQ(t4.dimension(), r4.dimension());
  for (typename T::dim_type i = 0; i < t4.dimension(); ++i) {
    EXPECT_EQ(t4.size(i), r4.size(i));
  }
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    expectEq(::std::real(c4) * u4(begin.position()), *begin);
    expectEq(-::std::real(s4) * u4(begin.position()), r4(begin.position()));
  }
  t4.copy(u4);
  r4.resize({20, 7}, {25, 3});
  for (typename T::reference_iterator begin = r4.reference_begin(),
           end = r4.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T v4 = r4.clone();
  r4 = linalg.rot(t4, r4, c4, s4);
  EXPECT_EQ(t4.dimension(), r4.dimension());
  for (typename T::dim_type i = 0; i < t4.dimension(); ++i) {
    EXPECT_EQ(t4.size(i), r4.size(i));
  }
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    D x4 = ::std::real(c4) * u4(begin.position()) +
        ::std::real(s4) * v4(begin.position());
    D y4 = ::std::real(c4) * v4(begin.position()) -
        ::std::real(s4) * u4(begin.position());
    expectEq(x4, *begin);
    expectEq(y4, r4(begin.position()));
  }

  T t5(10, 20, 7);
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  D c5 = rand(&gen, &dist);
  D s5 = rand(&gen, &dist);
  T u5 = t5.clone();
  T r5 = linalg.rot(t5, c5, s5);
  EXPECT_EQ(t5.dimension(), r5.dimension());
  for (typename T::dim_type i = 0; i < t5.dimension(); ++i) {
    EXPECT_EQ(t5.size(i), r5.size(i));
  }
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    expectEq(::std::real(c5) * u5(begin.position()), *begin);
    expectEq(-::std::real(s5) * u5(begin.position()), r5(begin.position()));
  }
  t5.copy(u5);
  r5.resize(10, 20, 7);
  for (typename T::reference_iterator begin = r5.reference_begin(),
           end = r5.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T v5 = r5.clone();
  r5 = linalg.rot(t5, r5, c5, s5);
  EXPECT_EQ(t5.dimension(), r5.dimension());
  for (typename T::dim_type i = 0; i < t5.dimension(); ++i) {
    EXPECT_EQ(t5.size(i), r5.size(i));
  }
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    D x5 = ::std::real(c5) * u5(begin.position()) +
        ::std::real(s5) * v5(begin.position());
    D y5 = ::std::real(c5) * v5(begin.position()) -
        ::std::real(s5) * u5(begin.position());
    expectEq(x5, *begin);
    expectEq(y5, r5(begin.position()));
  }

  T t6({10, 20, 7}, {309, 15, 2});
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  D c6 = rand(&gen, &dist);
  D s6 = rand(&gen, &dist);
  T u6 = t6.clone();
  T r6 = linalg.rot(t6, c6, s6);
  EXPECT_EQ(t6.dimension(), r6.dimension());
  for (typename T::dim_type i = 0; i < t6.dimension(); ++i) {
    EXPECT_EQ(t6.size(i), r6.size(i));
  }
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    expectEq(::std::real(c6) * u6(begin.position()), *begin);
    expectEq(-::std::real(s6) * u6(begin.position()), r6(begin.position()));
  }
  t6.copy(u6);
  r6.resize({10, 20, 7}, {365, 18, 2});
  for (typename T::reference_iterator begin = r6.reference_begin(),
           end = r6.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T v6 = r6.clone();
  r6 = linalg.rot(t6, r6, c6, s6);
  EXPECT_EQ(t6.dimension(), r6.dimension());
  for (typename T::dim_type i = 0; i < t6.dimension(); ++i) {
    EXPECT_EQ(t6.size(i), r6.size(i));
  }
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    D x6 = ::std::real(c6) * u6(begin.position()) +
        ::std::real(s6) * v6(begin.position());
    D y6 = ::std::real(c6) * v6(begin.position()) -
        ::std::real(s6) * u6(begin.position());
    expectEq(x6, *begin);
    expectEq(y6, r6(begin.position()));
  }
}

TEST(LinalgTest, rotTest) {
  rotTest< DoubleTensor >();
  rotTest< FloatTensor >();
  rotTest< DoubleComplexTensor >();
  rotTest< FloatComplexTensor >();
}

template < typename T >
void rotmTest() {
  typedef typename T::value_type D;
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  Linalg< T > linalg;

  T t1(7);
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T p1 = T(5);
  p1(0) = -1.0;
  for (int i = 1; i < 5; ++i) {
    p1(i) = rand(&gen, &dist);
  }
  T u1 = t1.clone();
  T r1 = linalg.rotm(t1, p1);
  EXPECT_EQ(t1.dimension(), r1.dimension());
  for (typename T::dim_type i = 0; i < t1.dimension(); ++i) {
    EXPECT_EQ(t1.size(i), r1.size(i));
  }
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    D x1 = p1(1) * u1(begin.position());
    D y1 = p1(2) * u1(begin.position());
    EXPECT_FLOAT_EQ(x1, *begin);
    EXPECT_FLOAT_EQ(y1, r1(begin.position()));
  }
  t1.copy(u1);
  r1.resize(7);
  for (typename T::reference_iterator begin = r1.reference_begin(),
           end = r1.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T v1 = r1.clone();
  r1 = linalg.rotm(t1, r1, p1);
  EXPECT_EQ(t1.dimension(), r1.dimension());
  for (typename T::dim_type i = 0; i < t1.dimension(); ++i) {
    EXPECT_EQ(t1.size(i), r1.size(i));
  }
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    D x1 = p1(1) * u1(begin.position()) + p1(3) * v1(begin.position());
    D y1 = p1(2) * u1(begin.position()) + p1(4) * v1(begin.position());
    EXPECT_FLOAT_EQ(x1, *begin);
    EXPECT_FLOAT_EQ(y1, r1(begin.position()));
  }

  T t2(typename T::size_storage({7}), typename T::stride_storage({2}));
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T p2 = T(5);
  p2(0) = -1.0;
  for (int i = 1; i < 5; ++i) {
    p2(i) = rand(&gen, &dist);
  }
  T u2 = t2.clone();
  T r2 = linalg.rotm(t2, p2);
  EXPECT_EQ(t2.dimension(), r2.dimension());
  for (typename T::dim_type i = 0; i < t2.dimension(); ++i) {
    EXPECT_EQ(t2.size(i), r2.size(i));
  }
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    D x2 = p2(1) * u2(begin.position());
    D y2 = p2(2) * u2(begin.position());
    EXPECT_FLOAT_EQ(x2, *begin);
    EXPECT_FLOAT_EQ(y2, r2(begin.position()));
  }
  t2.copy(u2);
  r2.resize(
      typename T::size_storage({7}), typename T::stride_storage({3}));
  for (typename T::reference_iterator begin = r2.reference_begin(),
           end = r2.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T v2 = r2.clone();
  r2 = linalg.rotm(t2, r2, p2);
  EXPECT_EQ(t2.dimension(), r2.dimension());
  for (typename T::dim_type i = 0; i < t2.dimension(); ++i) {
    EXPECT_EQ(t2.size(i), r2.size(i));
  }
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    D x2 = p2(1) * u2(begin.position()) + p2(3) * v2(begin.position());
    D y2 = p2(2) * u2(begin.position()) + p2(4) * v2(begin.position());
    EXPECT_FLOAT_EQ(x2, *begin);
    EXPECT_FLOAT_EQ(y2, r2(begin.position()));
  }

  T t3(20, 7);
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T p3 = T(5);
  p3(0) = -1.0;
  for (int i = 1; i < 5; ++i) {
    p3(i) = rand(&gen, &dist);
  }
  T u3 = t3.clone();
  T r3 = linalg.rotm(t3, p3);
  EXPECT_EQ(t3.dimension(), r3.dimension());
  for (typename T::dim_type i = 0; i < t3.dimension(); ++i) {
    EXPECT_EQ(t3.size(i), r3.size(i));
  }
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    D x3 = p3(1) * u3(begin.position());
    D y3 = p3(2) * u3(begin.position());
    EXPECT_FLOAT_EQ(x3, *begin);
    EXPECT_FLOAT_EQ(y3, r3(begin.position()));
  }
  t3.copy(u3);
  r3.resize(20, 7);
  for (typename T::reference_iterator begin = r3.reference_begin(),
           end = r3.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T v3 = r3.clone();
  r3 = linalg.rotm(t3, r3, p3);
  EXPECT_EQ(t3.dimension(), r3.dimension());
  for (typename T::dim_type i = 0; i < t3.dimension(); ++i) {
    EXPECT_EQ(t3.size(i), r3.size(i));
  }
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    D x3 = p3(1) * u3(begin.position()) + p3(3) * v3(begin.position());
    D y3 = p3(2) * u3(begin.position()) + p3(4) * v3(begin.position());
    EXPECT_FLOAT_EQ(x3, *begin);
    EXPECT_FLOAT_EQ(y3, r3(begin.position()));
  }

  T t4({20, 7}, {15, 2});
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T p4 = T(5);
  p4(0) = -1.0;
  for (int i = 1; i < 5; ++i) {
    p4(i) = rand(&gen, &dist);
  }
  T u4 = t4.clone();
  T r4 = linalg.rotm(t4, p4);
  EXPECT_EQ(t4.dimension(), r4.dimension());
  for (typename T::dim_type i = 0; i < t4.dimension(); ++i) {
    EXPECT_EQ(t4.size(i), r4.size(i));
  }
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    D x4 = p4(1) * u4(begin.position());
    D y4 = p4(2) * u4(begin.position());
    EXPECT_FLOAT_EQ(x4, *begin);
    EXPECT_FLOAT_EQ(y4, r4(begin.position()));
  }
  t4.copy(u4);
  r4.resize({20, 7}, {25, 3});
  for (typename T::reference_iterator begin = r4.reference_begin(),
           end = r4.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T v4 = r4.clone();
  r4 = linalg.rotm(t4, r4, p4);
  EXPECT_EQ(t4.dimension(), r4.dimension());
  for (typename T::dim_type i = 0; i < t4.dimension(); ++i) {
    EXPECT_EQ(t4.size(i), r4.size(i));
  }
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    D x4 = p4(1) * u4(begin.position()) + p4(3) * v4(begin.position());
    D y4 = p4(2) * u4(begin.position()) + p4(4) * v4(begin.position());
    EXPECT_FLOAT_EQ(x4, *begin);
    EXPECT_FLOAT_EQ(y4, r4(begin.position()));
  }

  T t5(10, 20, 7);
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T p5 = T(5);
  p5(0) = -1.0;
  for (int i = 1; i < 5; ++i) {
    p5(i) = rand(&gen, &dist);
  }
  T u5 = t5.clone();
  T r5 = linalg.rotm(t5, p5);
  EXPECT_EQ(t5.dimension(), r5.dimension());
  for (typename T::dim_type i = 0; i < t5.dimension(); ++i) {
    EXPECT_EQ(t5.size(i), r5.size(i));
  }
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    D x5 = p5(1) * u5(begin.position());
    D y5 = p5(2) * u5(begin.position());
    EXPECT_FLOAT_EQ(x5, *begin);
    EXPECT_FLOAT_EQ(y5, r5(begin.position()));
  }
  t5.copy(u5);
  r5.resize(10, 20, 7);
  for (typename T::reference_iterator begin = r5.reference_begin(),
           end = r5.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T v5 = r5.clone();
  r5 = linalg.rotm(t5, r5, p5);
  EXPECT_EQ(t5.dimension(), r5.dimension());
  for (typename T::dim_type i = 0; i < t5.dimension(); ++i) {
    EXPECT_EQ(t5.size(i), r5.size(i));
  }
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    D x5 = p5(1) * u5(begin.position()) + p5(3) * v5(begin.position());
    D y5 = p5(2) * u5(begin.position()) + p5(4) * v5(begin.position());
    EXPECT_FLOAT_EQ(x5, *begin);
    EXPECT_FLOAT_EQ(y5, r5(begin.position()));
  }

  T t6({10, 20, 7}, {309, 15, 2});
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T p6 = T(5);
  p6(0) = -1.0;
  for (int i = 1; i < 5; ++i) {
    p6(i) = rand(&gen, &dist);
  }
  T u6 = t6.clone();
  T r6 = linalg.rotm(t6, p6);
  EXPECT_EQ(t6.dimension(), r6.dimension());
  for (typename T::dim_type i = 0; i < t6.dimension(); ++i) {
    EXPECT_EQ(t6.size(i), r6.size(i));
  }
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    D x6 = p6(1) * u6(begin.position());
    D y6 = p6(2) * u6(begin.position());
    EXPECT_FLOAT_EQ(x6, *begin);
    EXPECT_FLOAT_EQ(y6, r6(begin.position()));
  }
  t6.copy(u6);
  r6.resize({10, 20, 7}, {365, 18, 2});
  for (typename T::reference_iterator begin = r6.reference_begin(),
           end = r6.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T v6 = r6.clone();
  r6 = linalg.rotm(t6, r6, p6);
  EXPECT_EQ(t6.dimension(), r6.dimension());
  for (typename T::dim_type i = 0; i < t6.dimension(); ++i) {
    EXPECT_EQ(t6.size(i), r6.size(i));
  }
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    D x6 = p6(1) * u6(begin.position()) + p6(3) * v6(begin.position());
    D y6 = p6(2) * u6(begin.position()) + p6(4) * v6(begin.position());
    EXPECT_FLOAT_EQ(x6, *begin);
    EXPECT_FLOAT_EQ(y6, r6(begin.position()));
  }
}

TEST(LinalgTest, rotmTest) {
  rotmTest< DoubleTensor >();
  rotmTest< FloatTensor >();
}

template < typename T >
void scalTest() {
  typedef typename T::value_type D;
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  Linalg< T > linalg;

  T t1(7);
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  D a1 = rand(&gen, &dist);
  T r1 = linalg.scal(t1.clone(), a1);
  EXPECT_EQ(t1.dimension(), r1.dimension());
  for (typename T::dim_type i = 0; i < t1.dimension(); ++i) {
    EXPECT_EQ(t1.size(i), r1.size(i));
  }
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    expectEq(a1 * (*begin), r1(begin.position()));
  }

  T t2(typename T::size_storage({7}), typename T::stride_storage({2}));
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  D a2 = rand(&gen, &dist);
  T r2 = linalg.scal(t2.clone(), a2);
  EXPECT_EQ(t2.dimension(), r2.dimension());
  for (typename T::dim_type i = 0; i < t2.dimension(); ++i) {
    EXPECT_EQ(t2.size(i), r2.size(i));
  }
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    expectEq(a2 * (*begin), r2(begin.position()));
  }

  T t3(20, 7);
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  D a3 = rand(&gen, &dist);
  T r3 = linalg.scal(t3.clone(), a3);
  EXPECT_EQ(t3.dimension(), r3.dimension());
  for (typename T::dim_type i = 0; i < t3.dimension(); ++i) {
    EXPECT_EQ(t3.size(i), r3.size(i));
  }
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    expectEq(a3 * (*begin), r3(begin.position()));
  }

  T t4({20, 7}, {15, 2});
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  D a4 = rand(&gen, &dist);
  T r4 = linalg.scal(t4.clone(), a4);
  EXPECT_EQ(t4.dimension(), r4.dimension());
  for (typename T::dim_type i = 0; i < t4.dimension(); ++i) {
    EXPECT_EQ(t4.size(i), r4.size(i));
  }
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    expectEq(a4 * (*begin), r4(begin.position()));
  }

  T t5(10, 20, 7);
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  D a5 = rand(&gen, &dist);
  T r5 = linalg.scal(t5.clone(), a5);
  EXPECT_EQ(t5.dimension(), r5.dimension());
  for (typename T::dim_type i = 0; i < t5.dimension(); ++i) {
    EXPECT_EQ(t5.size(i), r5.size(i));
  }
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    expectEq(a5 * (*begin), r5(begin.position()));
  }

  T t6({10, 20, 7}, {309, 15, 2});
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  D a6 = rand(&gen, &dist);
  T r6 = linalg.scal(t6.clone(), a6);
  EXPECT_EQ(t6.dimension(), r6.dimension());
  for (typename T::dim_type i = 0; i < t6.dimension(); ++i) {
    EXPECT_EQ(t6.size(i), r6.size(i));
  }
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    expectEq(a6 * (*begin), r6(begin.position()));
  }
}

TEST(LinalgTest, scalTest) {
  scalTest< DoubleTensor >();
  scalTest< FloatTensor >();
  scalTest< DoubleComplexTensor >();
  scalTest< FloatComplexTensor >();
}

template < typename T >
void swapTest() {
  typedef typename T::value_type D;
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  Linalg< T > linalg;

  T t1(7);
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T u1 = t1.clone();
  T r1 = linalg.swap(t1);
  EXPECT_EQ(t1.dimension(), r1.dimension());
  for (typename T::dim_type i = 0; i < t1.dimension(); ++i) {
    EXPECT_EQ(t1.size(i), r1.size(i));
  }
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    expectEq(D(0), *begin);
    expectEq(u1(begin.position()), r1(begin.position()));
  }
  t1.copy(u1);
  r1.resize(7);
  for (typename T::reference_iterator begin = r1.reference_begin(),
           end = r1.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T v1 = r1.clone();
  r1 = linalg.swap(t1, r1);
  EXPECT_EQ(t1.dimension(), r1.dimension());
  for (typename T::dim_type i = 0; i < t1.dimension(); ++i) {
    EXPECT_EQ(t1.size(i), r1.size(i));
  }
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    expectEq(v1(begin.position()), *begin);
    expectEq(u1(begin.position()), r1(begin.position()));
  }

  T t2(typename T::size_storage({7}), typename T::stride_storage({2}));
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T u2 = t2.clone();
  T r2 = linalg.swap(t2);
  EXPECT_EQ(t2.dimension(), r2.dimension());
  for (typename T::dim_type i = 0; i < t2.dimension(); ++i) {
    EXPECT_EQ(t2.size(i), r2.size(i));
  }
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    expectEq(D(0), *begin);
    expectEq(u2(begin.position()), r2(begin.position()));
  }
  t2.copy(u2);
  r2.resize(typename T::size_storage({7}), typename T::stride_storage({3}));
  for (typename T::reference_iterator begin = r2.reference_begin(),
           end = r2.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T v2 = r2.clone();
  r2 = linalg.swap(t2, r2);
  EXPECT_EQ(t2.dimension(), r2.dimension());
  for (typename T::dim_type i = 0; i < t2.dimension(); ++i) {
    EXPECT_EQ(t2.size(i), r2.size(i));
  }
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    expectEq(v2(begin.position()), *begin);
    expectEq(u2(begin.position()), r2(begin.position()));
  }

  T t3(20, 7);
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T u3 = t3.clone();
  T r3 = linalg.swap(t3);
  EXPECT_EQ(t3.dimension(), r3.dimension());
  for (typename T::dim_type i = 0; i < t3.dimension(); ++i) {
    EXPECT_EQ(t3.size(i), r3.size(i));
  }
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    expectEq(D(0), *begin);
    expectEq(u3(begin.position()), r3(begin.position()));
  }
  t3.copy(u3);
  r3.resize(20, 7);
  for (typename T::reference_iterator begin = r3.reference_begin(),
           end = r3.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T v3 = r3.clone();
  r3 = linalg.swap(t3, r3);
  EXPECT_EQ(t3.dimension(), r3.dimension());
  for (typename T::dim_type i = 0; i < t3.dimension(); ++i) {
    EXPECT_EQ(t3.size(i), r3.size(i));
  }
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    expectEq(v3(begin.position()), *begin);
    expectEq(u3(begin.position()), r3(begin.position()));
  }

  T t4({20, 7}, {15, 2});
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T u4 = t4.clone();
  T r4 = linalg.swap(t4);
  EXPECT_EQ(t4.dimension(), r4.dimension());
  for (typename T::dim_type i = 0; i < t4.dimension(); ++i) {
    EXPECT_EQ(t4.size(i), r4.size(i));
  }
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    expectEq(D(0), *begin);
    expectEq(u4(begin.position()), r4(begin.position()));
  }
  t4.copy(u4);
  r4.resize({20, 7}, {25, 3});
  for (typename T::reference_iterator begin = r4.reference_begin(),
           end = r4.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T v4 = r4.clone();
  r4 = linalg.swap(t4, r4);
  EXPECT_EQ(t4.dimension(), r4.dimension());
  for (typename T::dim_type i = 0; i < t4.dimension(); ++i) {
    EXPECT_EQ(t4.size(i), r4.size(i));
  }
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    expectEq(v4(begin.position()), *begin);
    expectEq(u4(begin.position()), r4(begin.position()));
  }

  T t5(10, 20, 7);
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T u5 = t5.clone();
  T r5 = linalg.swap(t5);
  EXPECT_EQ(t5.dimension(), r5.dimension());
  for (typename T::dim_type i = 0; i < t5.dimension(); ++i) {
    EXPECT_EQ(t5.size(i), r5.size(i));
  }
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    expectEq(D(0), *begin);
    expectEq(u5(begin.position()), r5(begin.position()));
  }
  t5.copy(u5);
  r5.resize(10, 20, 7);
  for (typename T::reference_iterator begin = r5.reference_begin(),
           end = r5.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T v5 = r5.clone();
  r5 = linalg.swap(t5, r5);
  EXPECT_EQ(t5.dimension(), r5.dimension());
  for (typename T::dim_type i = 0; i < t5.dimension(); ++i) {
    EXPECT_EQ(t5.size(i), r5.size(i));
  }
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    expectEq(v5(begin.position()), *begin);
    expectEq(u5(begin.position()), r5(begin.position()));
  }

  T t6({10, 20, 7}, {309, 15, 2});
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T u6 = t6.clone();
  T r6 = linalg.swap(t6);
  EXPECT_EQ(t6.dimension(), r6.dimension());
  for (typename T::dim_type i = 0; i < t6.dimension(); ++i) {
    EXPECT_EQ(t6.size(i), r6.size(i));
  }
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    expectEq(D(0), *begin);
    expectEq(u6(begin.position()), r6(begin.position()));
  }
  t6.copy(u6);
  r6.resize({10, 20, 7}, {365, 18, 2});
  for (typename T::reference_iterator begin = r6.reference_begin(),
           end = r6.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  T v6 = r6.clone();
  r6 = linalg.swap(t6, r6);
  EXPECT_EQ(t6.dimension(), r6.dimension());
  for (typename T::dim_type i = 0; i < t6.dimension(); ++i) {
    EXPECT_EQ(t6.size(i), r6.size(i));
  }
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    expectEq(v6(begin.position()), *begin);
    expectEq(u6(begin.position()), r6(begin.position()));
  }
}

TEST(LinalgTest, swapTest) {
  swapTest< DoubleTensor >();
  swapTest< FloatTensor >();
  swapTest< DoubleComplexTensor >();
  swapTest< FloatComplexTensor >();
}

template < typename T >
void iamaxTest() {
  typedef typename T::value_type D;
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  Linalg< T > linalg;

  T t1(7);
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  SizeTensor r1 = linalg.iamax(t1);
  int s1 = cxxblas::iamax(t1.size(0), t1.data(), t1.stride(0));
  EXPECT_EQ(1, r1.dimension());
  EXPECT_EQ(1, r1.size(0));
  EXPECT_EQ(s1, r1(0));

  T t2(typename T::size_storage({7}), typename T::stride_storage({2}));
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  SizeTensor r2 = linalg.iamax(t2);
  int s2 = cxxblas::iamax(t2.size(0), t2.data(), t2.stride(0));
  EXPECT_EQ(1, r2.dimension());
  EXPECT_EQ(1, r2.size(0));
  EXPECT_EQ(s2, r2(0));

  T t3(20, 7);
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  SizeTensor r3 = linalg.iamax(t3);
  EXPECT_EQ(1, r3.dimension());
  EXPECT_EQ(20, r3.size(0));
  for (int i = 0; i < 20; ++i) {
    int s3 = cxxblas::iamax(t3[i].size(0), t3[i].data(), t3[i].stride(0));
    EXPECT_EQ(s3, r3(i));
  }

  T t4({20, 7}, {14, 2});
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  SizeTensor r4 = linalg.iamax(t4);
  EXPECT_EQ(1, r4.dimension());
  EXPECT_EQ(20, r4.size(0));
  for (int i = 0; i < 20; ++i) {
    int s4 = cxxblas::iamax(t4[i].size(0), t4[i].data(), t4[i].stride(0));
    EXPECT_EQ(s4, r4(i));
  }

  T t5(10, 20, 7);
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  SizeTensor r5 = linalg.iamax(t5);
  EXPECT_EQ(2, r5.dimension());
  EXPECT_EQ(10, r5.size(0));
  EXPECT_EQ(20, r5.size(1));
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 20; ++j) {
      int s5 = cxxblas::iamax(
          t5[i][j].size(0), t5[i][j].data(), t5[i][j].stride(0));
      EXPECT_EQ(s5, r5(i, j));
    }
  }

  T t6({10, 20, 7}, {302, 15, 2});
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    *begin = rand(&gen, &dist);
  }
  SizeTensor r6 = linalg.iamax(t6);
  EXPECT_EQ(2, r6.dimension());
  EXPECT_EQ(10, r6.size(0));
  EXPECT_EQ(20, r6.size(1));
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 20; ++j) {
      int s6 = cxxblas::iamax(
          t6[i][j].size(0), t6[i][j].data(), t6[i][j].stride(0));
      EXPECT_EQ(s6, r6(i, j));
    }
  }
}

TEST(LinalgTest, iamaxTest) {
  iamaxTest< DoubleTensor >();
  iamaxTest< FloatTensor >();
  iamaxTest< DoubleComplexTensor >();
  iamaxTest< FloatComplexTensor >();
}

}  // namespace
}  // namespace linalg
}  // namespace thunder
