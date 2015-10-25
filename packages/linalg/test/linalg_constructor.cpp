/*
 * \copyright Copyright 2014-2015 Xiang Zhang All Rights Reserved.
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

#include <cmath>
#include <complex>
#include <cstdlib>
#include <random>

#include "gtest/gtest.h"
#include "thunder/linalg.hpp"
#include "thunder/linalg/math.hpp"
#include "thunder/linalg/linalg.hpp"
#include "thunder/random.hpp"
#include "thunder/tensor.hpp"
#include "thunder/storage.hpp"

#include "thunder/linalg/linalg-inl.hpp"
#include "thunder/linalg/math-inl.hpp"

namespace thunder {
namespace linalg {
namespace {

#define EXPECT_COMPLEX_EQ(expected, actual)                     \
  EXPECT_FLOAT_EQ(::std::real(expected), ::std::real(actual));  \
  EXPECT_FLOAT_EQ(::std::imag(expected), ::std::imag(actual));

template < typename T >
T randomConstruct(T t) {
  Random< T > random;
  return random.uniform(t);
}
template < typename D >
Tensor< Storage< ::std::complex< D > > > randomConstruct(
    Tensor< Storage< ::std::complex< D > > > t) {
  typedef ::std::complex< D > C;

  ::std::mt19937 generator;
  ::std::uniform_real_distribution< D > random(0.0, 1.0);
  for (auto begin = t.reference_begin(), end = t.reference_end();
       begin != end; ++begin) {
    (*begin) = C(random(generator), random(generator));
  }
  return t;
}

template < typename T >
void diagTest() {
  Linalg< T > linalg;

  // Give a vector, construct a new matrix
  T t1 = randomConstruct(T(10));
  T r1 = linalg.diag(t1);
  EXPECT_EQ(2, r1.dimension());
  EXPECT_EQ(t1.size(0), r1.size(0));
  EXPECT_EQ(t1.size(0), r1.size(1));
  for (int i = 0; i < t1.size(0); ++i) {
    for (int j = 0; j < t1.size(0); ++j) {
      if (i == j) {
        EXPECT_COMPLEX_EQ(t1(i), r1(i, j));
      } else {
        EXPECT_COMPLEX_EQ(0, r1(i, j));
      }
    }
  }

  // Give a matrix, construct a new vector
  T t2 = randomConstruct(T(10, 13));
  T r2 = linalg.diag(t2);
  EXPECT_EQ(1, r2.dimension());
  EXPECT_EQ(::std::min(t2.size(0), t2.size(1)), r2.size(0));
  for (int i = 0; i < ::std::min(t2.size(0), t2.size(1)); ++i) {
    EXPECT_COMPLEX_EQ(t2(i, i), r2(i));
  }

  // Give a multi-dimensional array, construct vectors in batch mode
  T t3 = randomConstruct(T(10, 13, 7, 4));
  T r3 = linalg.diag(t3);
  EXPECT_EQ(t3.dimension() - 1, r3.dimension());
  for (int i = 0; i < t3.dimension() - 2; ++i) {
    EXPECT_EQ(t3.size(i), r3.size(i));
  }
  EXPECT_EQ(
      ::std::min(t3.size(t3.dimension() - 2), t3.size(t3.dimension() - 1)),
      r3.size(r3.dimension() - 1));
  T r3_sub = r3.select(r3.dimension() - 1, 0);
  for (auto begin = r3_sub.reference_begin(), end = r3_sub.reference_end();
       begin != end; ++begin) {
    T matrix = t3[begin.position()];
    T vector = r3[begin.position()];
    for (int i = 0; i < vector.size(0); ++i) {
      EXPECT_COMPLEX_EQ(matrix(i, i), vector(i));
    }
  }

  // Give a non-conitugous multi-dimensional array, construct vectors
  T t4 = randomConstruct(T({10, 13, 7, 4}, {970, 73, 10, 2}));
  T r4 = linalg.diag(t4);
  EXPECT_EQ(t4.dimension() - 1, r4.dimension());
  for (int i = 0; i < t4.dimension() - 2; ++i) {
    EXPECT_EQ(t4.size(i), r4.size(i));
  }
  EXPECT_EQ(
      ::std::min(t4.size(t4.dimension() - 2), t4.size(t4.dimension() - 1)),
      r4.size(r4.dimension() - 1));
  T r4_sub = r4.select(r4.dimension() - 1, 0);
  for (auto begin = r4_sub.reference_begin(), end = r4_sub.reference_end();
       begin != end; ++begin) {
    T matrix = t4[begin.position()];
    T vector = r4[begin.position()];
    for (int i = 0; i < vector.size(0); ++i) {
      EXPECT_COMPLEX_EQ(matrix(i, i), vector(i));
    }
  }
}

TEST(LinalgTest, diagTest) {
  diagTest< DoubleTensor >();
  diagTest< FloatTensor >();
  diagTest< DoubleComplexTensor >();
  diagTest< FloatComplexTensor >();
}

template < typename T >
void eyeTest() {
  Linalg< T > linalg;

  // Dimension 1 will produce a square matrix
  SizeStorage s1(1);
  s1[0] = 5;
  T r1 = linalg.eye(s1);
  EXPECT_EQ(2, r1.dimension());
  EXPECT_EQ(s1[0], r1.size(0));
  EXPECT_EQ(s1[0], r1.size(1));
  for (int i = 0; i < s1[0]; ++i) {
    for (int j = 0; j < s1[0]; ++j) {
      if (i == j) {
        EXPECT_COMPLEX_EQ(1, r1(i, j));
      } else {
        EXPECT_COMPLEX_EQ(0, r1(i, j));
      }
    }
  }

  // Dimension 2 will produce matrices of that size
  SizeStorage s2({5, 8});
  T r2 = linalg.eye(s2);
  EXPECT_EQ(s2.size(), r2.dimension());
  for (int i = 0; i < s2.size(); ++i) {
    EXPECT_EQ(s2[i], r2.size(i));
  }
  for (int i = 0; i < s2[0]; ++i) {
    for (int j = 0; j < s2[1]; ++j) {
      if (i == j) {
        EXPECT_COMPLEX_EQ(1, r2(i, j));
      } else {
        EXPECT_COMPLEX_EQ(0, r2(i, j));
      }
    }
  }

  // Contiguous batch mode
  T r3 = linalg.eye({10, 13, 7, 4});
  T r3_sub = r3.select(r3.dimension() - 1, 0).select(r3.dimension() - 2, 0);
  for (auto begin = r3_sub.reference_begin(), end = r3_sub.reference_end();
       begin != end; ++begin) {
    T matrix = r3[begin.position()];
    for (int i = 0; i < matrix.size(0); ++i) {
      for (int j = 0; j < matrix.size(1); ++j) {
        if (i == j) {
          EXPECT_COMPLEX_EQ(1, matrix(i, j));
        } else {
          EXPECT_COMPLEX_EQ(0, matrix(i, j));
        }
      }
    }
  }

  // Non-contiguous batch mode
  T r4({10, 13, 7, 4}, {970, 73, 10, 2});
  r4 = linalg.eye(r4.size(), r4);
  T r4_sub = r4.select(r4.dimension() - 1, 0).select(r4.dimension() - 2, 0);
  for (auto begin = r4_sub.reference_begin(), end = r4_sub.reference_end();
       begin != end; ++begin) {
    T matrix = r4[begin.position()];
    for (int i = 0; i < matrix.size(0); ++i) {
      for (int j = 0; j < matrix.size(1); ++j) {
        if (i == j) {
          EXPECT_COMPLEX_EQ(1, matrix(i, j));
        } else {
          EXPECT_COMPLEX_EQ(0, matrix(i, j));
        }
      }
    }
  }
}

TEST(LinalgTest, eyeTest) {
  eyeTest< DoubleTensor >();
  eyeTest< FloatTensor >();
  eyeTest< DoubleComplexTensor >();
  eyeTest< FloatComplexTensor >();
}

template < typename T >
void linspaceTest() {
  typedef typename T::value_type D;
  Linalg< T > linalg;

  // Default call gives 100 values
  T r1 = linalg.linspace(1, 100);
  EXPECT_EQ(1, r1.dimension());
  EXPECT_EQ(100, r1.size(0));
  D i1 = (D(100) - D(1)) / static_cast< D >(r1.size(0) - 1);
  for (int i = 1; i < r1.size(0); ++i) {
    EXPECT_COMPLEX_EQ(i1, r1(i) - r1(i - 1));
  }

  // Now get 10 values
  T r2 = linalg.linspace(10, 100, 10);
  EXPECT_EQ(1, r2.dimension());
  EXPECT_EQ(10, r2.size(0));
  D i2 = (D(100) - D(10)) / static_cast< D >(r2.size(0) - 1);
  for (int i = 1; i < r2.size(0); ++i) {
    EXPECT_COMPLEX_EQ(i2, r2(i) - r2(i - 1));
  }
}

TEST(LinalgTest, linspaceTest) {
  linspaceTest< DoubleTensor >();
  linspaceTest< FloatTensor >();
  linspaceTest< DoubleComplexTensor >();
  linspaceTest< FloatComplexTensor >();
}

template < typename T >
void logspaceTest() {
  typedef typename T::value_type D;
  Linalg< T > linalg;

  // Get 10 values
  T r1 = linalg.logspace(1, 10, 10);
  EXPECT_EQ(1, r1.dimension());
  EXPECT_EQ(10, r1.size(0));
  D i1 = ::std::pow(D(10), (D(10) - D(1)) / static_cast< D >(r1.size(0) - 1));
  for (int i = 1; i < r1.size(0); ++i) {
    EXPECT_COMPLEX_EQ(i1, r1(i) / r1(i - 1));
  }
}

TEST(LinalgTest, logspaceTest) {
  logspaceTest< DoubleTensor >();
  logspaceTest< FloatTensor >();
  logspaceTest< DoubleComplexTensor >();
  logspaceTest< FloatComplexTensor >();
}

template < typename T >
void trilTest() {
  Linalg< T > linalg;

  // Dimension is 2, normal lower triangular
  T t1 = randomConstruct(T(10, 13));
  T r1 = linalg.tril(t1);
  EXPECT_EQ(t1.dimension(), r1.dimension());
  EXPECT_EQ(t1.size(0), r1.size(0));
  EXPECT_EQ(t1.size(1), r1.size(1));
  for (int i = 0; i < t1.size(0); ++i) {
    for (int j = 0; j < t1.size(1); ++j) {
      if (i < j ) {
        EXPECT_COMPLEX_EQ(0, r1(i, j));
      } else {
        EXPECT_COMPLEX_EQ(t1(i, j), r1(i, j));
      }
    }
  }

  // Contiguous batch mode
  T t2 = randomConstruct(T(10, 13, 7, 4));
  T r2 = linalg.tril(t2);
  EXPECT_EQ(t2.dimension(), r2.dimension());
  for (int i = 0; i < t2.dimension(); ++i) {
    EXPECT_EQ(t2.size(i), r2.size(i));
  }
  T t2_sub = t2.select(t2.dimension() - 1, 0).select(t2.dimension() - 2, 0);
  for (auto begin = t2_sub.reference_begin(), end = t2_sub.reference_end();
       begin != end; ++begin) {
    T t2_matrix = t2[begin.position()];
    T r2_matrix = r2[begin.position()];
    for (int i = 0; i < t2.size(t2.dimension() - 2); ++i) {
      for (int j = 0; j < t2.size(t2.dimension() - 1); ++j) {
        if (i < j) {
          EXPECT_COMPLEX_EQ(0, r2_matrix(i, j));
        } else {
          EXPECT_COMPLEX_EQ(t2_matrix(i, j), r2_matrix(i, j));
        }
      }
    }
  }

  // Noncontiguous batch mode
  T t3 = randomConstruct(T({10, 13, 7, 4}, {970, 73, 10, 2}));
  T r3 = linalg.tril(t3);
  EXPECT_EQ(t3.dimension(), r3.dimension());
  for (int i = 0; i < t3.dimension(); ++i) {
    EXPECT_EQ(t3.size(i), r3.size(i));
  }
  T t3_sub = t3.select(t3.dimension() - 1, 0).select(t3.dimension() - 2, 0);
  for (auto begin = t3_sub.reference_begin(), end = t3_sub.reference_end();
       begin != end; ++begin) {
    T t3_matrix = t3[begin.position()];
    T r3_matrix = r3[begin.position()];
    for (int i = 0; i < t3.size(t3.dimension() - 2); ++i) {
      for (int j = 0; j < t3.size(t3.dimension() - 1); ++j) {
        if (i < j) {
          EXPECT_COMPLEX_EQ(0, r3_matrix(i, j));
        } else {
          EXPECT_COMPLEX_EQ(t3_matrix(i, j), r3_matrix(i, j));
        }
      }
    }
  }
}

TEST(LinalgTest, trilTest) {
  trilTest< DoubleTensor >();
  trilTest< FloatTensor >();
  trilTest< DoubleComplexTensor >();
  trilTest< FloatComplexTensor >();
}

template < typename T >
void triuTest() {
  Linalg< T > linalg;

  // Dimension is 2, normal lower triangular
  T t1 = randomConstruct(T(10, 13));
  T r1 = linalg.triu(t1);
  EXPECT_EQ(t1.dimension(), r1.dimension());
  EXPECT_EQ(t1.size(0), r1.size(0));
  EXPECT_EQ(t1.size(1), r1.size(1));
  for (int i = 0; i < t1.size(0); ++i) {
    for (int j = 0; j < t1.size(1); ++j) {
      if (i > j ) {
        EXPECT_COMPLEX_EQ(0, r1(i, j));
      } else {
        EXPECT_COMPLEX_EQ(t1(i, j), r1(i, j));
      }
    }
  }

  // Contiguous batch mode
  T t2 = randomConstruct(T(10, 13, 7, 4));
  T r2 = linalg.triu(t2);
  EXPECT_EQ(t2.dimension(), r2.dimension());
  for (int i = 0; i < t2.dimension(); ++i) {
    EXPECT_EQ(t2.size(i), r2.size(i));
  }
  T t2_sub = t2.select(t2.dimension() - 1, 0).select(t2.dimension() - 2, 0);
  for (auto begin = t2_sub.reference_begin(), end = t2_sub.reference_end();
       begin != end; ++begin) {
    T t2_matrix = t2[begin.position()];
    T r2_matrix = r2[begin.position()];
    for (int i = 0; i < t2.size(t2.dimension() - 2); ++i) {
      for (int j = 0; j < t2.size(t2.dimension() - 1); ++j) {
        if (i > j) {
          EXPECT_COMPLEX_EQ(0, r2_matrix(i, j));
        } else {
          EXPECT_COMPLEX_EQ(t2_matrix(i, j), r2_matrix(i, j));
        }
      }
    }
  }

  // Noncontiguous batch mode
  T t3 = randomConstruct(T({10, 13, 7, 4}, {970, 73, 10, 2}));
  T r3 = linalg.triu(t3);
  EXPECT_EQ(t3.dimension(), r3.dimension());
  for (int i = 0; i < t3.dimension(); ++i) {
    EXPECT_EQ(t3.size(i), r3.size(i));
  }
  T t3_sub = t3.select(t3.dimension() - 1, 0).select(t3.dimension() - 2, 0);
  for (auto begin = t3_sub.reference_begin(), end = t3_sub.reference_end();
       begin != end; ++begin) {
    T t3_matrix = t3[begin.position()];
    T r3_matrix = r3[begin.position()];
    for (int i = 0; i < t3.size(t3.dimension() - 2); ++i) {
      for (int j = 0; j < t3.size(t3.dimension() - 1); ++j) {
        if (i > j) {
          EXPECT_COMPLEX_EQ(0, r3_matrix(i, j));
        } else {
          EXPECT_COMPLEX_EQ(t3_matrix(i, j), r3_matrix(i, j));
        }
      }
    }
  }
}

TEST(LinalgTest, triuTest) {
  triuTest< DoubleTensor >();
  triuTest< FloatTensor >();
  triuTest< DoubleComplexTensor >();
  triuTest< FloatComplexTensor >();
}

}  // namespace
}  // namespace linalg
}  // namespace thunder
