/*
 * \copyright Copyright 2015 Xiang Zhang All Rights Reserved.
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

#include <complex>
#include <cstdlib>
#include <random>

#include "gtest/gtest.h"
#include "thunder/linalg/cxxblas.hpp"

namespace thunder {
namespace linalg {
namespace {

TEST(CxxBlasTest, asumTest) {
  const int n = 1000;
  ::std::mt19937 gen;
  ::std::uniform_real_distribution< double > dist;

  float sx[n], ss = 0.0, sh = 0.0;
  for (int i = 0; i < n; ++i) {
    sx[i] = static_cast< float >(dist(gen));
    ss += ::std::fabs(sx[i]);
    if (i % 2 == 0) {
      sh += ::std::fabs(sx[i]);
    }
  }
  EXPECT_FLOAT_EQ(ss, cxxblas::asum(n, sx));
  EXPECT_FLOAT_EQ(sh, cxxblas::asum(n / 2, sx, 2));

  double dx[n], ds = 0.0, dh = 0.0;
  for (int i = 0; i < n; ++i) {
    dx[i] = static_cast< double >(dist(gen));
    ds += ::std::fabs(dx[i]);
    if (i % 2 == 0) {
      dh += ::std::fabs(dx[i]);
    }
  }
  EXPECT_FLOAT_EQ(ds, cxxblas::asum(n, dx));
  EXPECT_FLOAT_EQ(dh, cxxblas::asum(n / 2, dx, 2));

  ::std::complex< float > cx[n];
  float cs = 0.0, ch = 0.0;
  for (int i = 0; i < n; ++i) {
    cx[i] = ::std::complex< float >(dist(gen), dist(gen));
    cs += ::std::fabs(::std::real(cx[i])) + ::std::fabs(::std::imag(cx[i]));
    if (i % 2 == 0) {
      ch += ::std::fabs(::std::real(cx[i])) + ::std::fabs(::std::imag(cx[i]));
    }
  }
  EXPECT_FLOAT_EQ(cs, cxxblas::asum(n, cx));
  EXPECT_FLOAT_EQ(ch, cxxblas::asum(n / 2, cx, 2));

  ::std::complex< double > zx[n];
  double zs = 0.0, zh = 0.0;
  for (int i = 0; i < n; ++i) {
    zx[i] = ::std::complex< double >(dist(gen), dist(gen));
    zs += ::std::fabs(::std::real(zx[i])) + ::std::fabs(::std::imag(zx[i]));
    if (i % 2 == 0) {
      zh += ::std::fabs(::std::real(zx[i])) + ::std::fabs(::std::imag(zx[i]));
    }
  }
  EXPECT_FLOAT_EQ(zs, cxxblas::asum(n, zx));
  EXPECT_FLOAT_EQ(zh, cxxblas::asum(n / 2, zx, 2));
}

TEST(CxxBlasTest, axpyTest) {
  const int n = 1000;
  ::std::mt19937 gen;
  ::std::uniform_real_distribution< double > dist;

  float sx[n], sy[n], sr[n], sh[n];
  float sa = static_cast< float >(dist(gen));
  for (int i = 0; i < n; ++i) {
    sx[i] = static_cast< float >(dist(gen));
    sy[i] = static_cast< float >(dist(gen));
    sr[i] = sy[i];
    if (i % 2 == 0) {
      sh[i] = sy[i];
    } else {
      sh[i] = static_cast< float >(dist(gen));
    }
  }
  cxxblas::axpy(n, sx, sr);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(1.0 * sx[i] + sy[i], sr[i]);
  }
  cxxblas::axpy(n / 2, sx, sh, sa, 2, 2);
  for (int i = 0; i < n / 2; ++i) {
    EXPECT_FLOAT_EQ(sa * sx[i * 2] + sy[i * 2], sh[i * 2]);
  }

  double dx[n], dy[n], dr[n], dh[n];
  double da = static_cast< double >(dist(gen));
  for (int i = 0; i < n; ++i) {
    dx[i] = static_cast< double >(dist(gen));
    dy[i] = static_cast< double >(dist(gen));
    dr[i] = dy[i];
    if (i % 2 == 0) {
      dh[i] = dy[i];
    } else {
      dh[i] = static_cast< double >(dist(gen));
    }
  }
  cxxblas::axpy(n, dx, dr);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(1.0 * dx[i] + dy[i], dr[i]);
  }
  cxxblas::axpy(n / 2, dx, dh, da, 2, 2);
  for (int i = 0; i < n / 2; ++i) {
    EXPECT_FLOAT_EQ(da * dx[i * 2] + dy[i * 2], dh[i * 2]);
  }

  ::std::complex< float > cx[n], cy[n], cr[n], ch[n];
  ::std::complex< float > ca = ::std::complex< float >(dist(gen), dist(gen));
  for (int i = 0; i < n; ++i) {
    cx[i] = ::std::complex< float >(dist(gen), dist(gen));
    cy[i] = ::std::complex< float >(dist(gen), dist(gen));
    cr[i] = cy[i];
    if (i % 2 == 0) {
      ch[i] = cy[i];
    } else {
      ch[i] = ::std::complex< float >(dist(gen), dist(gen));
    }
  }
  cxxblas::axpy(n, cx, cr);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(::std::real(cx[i] + cy[i]), ::std::real(cr[i]));
    EXPECT_FLOAT_EQ(::std::imag(cx[i] + cy[i]), ::std::imag(cr[i]));
  }
  cxxblas::axpy(n / 2, cx, ch, ca, 2, 2);
  for (int i = 0; i < n / 2; ++i) {
    EXPECT_FLOAT_EQ(::std::real(ca * cx[i * 2] + cy[i * 2]),
                    ::std::real(ch[i * 2]));
    EXPECT_FLOAT_EQ(::std::imag(ca * cx[i * 2] + cy[i * 2]),
                    ::std::imag(ch[i * 2]));
  }

  ::std::complex< double > zx[n], zy[n], zr[n], zh[n];
  ::std::complex< double > za = ::std::complex< double >(dist(gen), dist(gen));
  for (int i = 0; i < n; ++i) {
    zx[i] = ::std::complex< double >(dist(gen), dist(gen));
    zy[i] = ::std::complex< double >(dist(gen), dist(gen));
    zr[i] = zy[i];
    if (i % 2 == 0) {
      zh[i] = zy[i];
    } else {
      zh[i] = ::std::complex< double >(dist(gen), dist(gen));
    }
  }
  cxxblas::axpy(n, zx, zr);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(::std::real(zx[i] + zy[i]), ::std::real(zr[i]));
    EXPECT_FLOAT_EQ(::std::imag(zx[i] + zy[i]), ::std::imag(zr[i]));
  }
  cxxblas::axpy(n / 2, zx, zh, za, 2, 2);
  for (int i = 0; i < n / 2; ++i) {
    EXPECT_FLOAT_EQ(::std::real(za * zx[i * 2] + zy[i * 2]),
                    ::std::real(zh[i * 2]));
    EXPECT_FLOAT_EQ(::std::imag(za * zx[i * 2] + zy[i * 2]),
                    ::std::imag(zh[i * 2]));
  }
}

TEST(CxxBlasTest, copyTest) {
  const int n = 1000;
  ::std::mt19937 gen;
  ::std::uniform_real_distribution< double > dist;

  float sx[n], sy[n], sr[n], sh[n];
  for (int i = 0; i < n; ++i) {
    sx[i] = static_cast< float >(dist(gen));
    sy[i] = static_cast< float >(dist(gen));
    sr[i] = sy[i];
    if (i % 2 == 0) {
      sh[i] = sy[i];
    } else {
      sh[i] = static_cast< float >(dist(gen));
    }
  }
  cxxblas::copy(n, sx, sr);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(sx[i], sr[i]);
  }
  cxxblas::copy(n / 2, sx, sh, 2, 2);
  for (int i = 0; i < n / 2; ++i) {
    EXPECT_FLOAT_EQ(sx[i * 2], sh[i * 2]);
  }

  double dx[n], dy[n], dr[n], dh[n];
  for (int i = 0; i < n; ++i) {
    dx[i] = static_cast< double >(dist(gen));
    dy[i] = static_cast< double >(dist(gen));
    dr[i] = dy[i];
    if (i % 2 == 0) {
      dh[i] = dy[i];
    } else {
      dh[i] = static_cast< double >(dist(gen));
    }
  }
  cxxblas::copy(n, dx, dr);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(dx[i], dr[i]);
  }
  cxxblas::copy(n / 2, dx, dh, 2, 2);
  for (int i = 0; i < n / 2; ++i) {
    EXPECT_FLOAT_EQ(dx[i * 2], dh[i * 2]);
  }

  ::std::complex< float > cx[n], cy[n], cr[n], ch[n];
  for (int i = 0; i < n; ++i) {
    cx[i] = ::std::complex< float >(dist(gen), dist(gen));
    cy[i] = ::std::complex< float >(dist(gen), dist(gen));
    cr[i] = cy[i];
    if (i % 2 == 0) {
      ch[i] = cy[i];
    } else {
      ch[i] = ::std::complex< float >(dist(gen), dist(gen));
    }
  }
  cxxblas::copy(n, cx, cr);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(::std::real(cx[i]), ::std::real(cr[i]));
    EXPECT_FLOAT_EQ(::std::imag(cx[i]), ::std::imag(cr[i]));
  }
  cxxblas::copy(n / 2, cx, ch, 2, 2);
  for (int i = 0; i < n / 2; ++i) {
    EXPECT_FLOAT_EQ(::std::real(cx[i * 2]), ::std::real(ch[i * 2]));
    EXPECT_FLOAT_EQ(::std::imag(cx[i * 2]), ::std::imag(ch[i * 2]));
  }

  ::std::complex< double > zx[n], zy[n], zr[n], zh[n];
  for (int i = 0; i < n; ++i) {
    zx[i] = ::std::complex< double >(dist(gen), dist(gen));
    zy[i] = ::std::complex< double >(dist(gen), dist(gen));
    zr[i] = zy[i];
    if (i % 2 == 0) {
      zh[i] = zy[i];
    } else {
      zh[i] = ::std::complex< double >(dist(gen), dist(gen));
    }
  }
  cxxblas::copy(n, zx, zr);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(::std::real(zx[i]), ::std::real(zr[i]));
    EXPECT_FLOAT_EQ(::std::imag(zx[i]), ::std::imag(zr[i]));
  }
  cxxblas::copy(n / 2, zx, zh, 2, 2);
  for (int i = 0; i < n / 2; ++i) {
    EXPECT_FLOAT_EQ(::std::real(zx[i * 2]), ::std::real(zh[i * 2]));
    EXPECT_FLOAT_EQ(::std::imag(zx[i * 2]), ::std::imag(zh[i * 2]));
  }
}

TEST(CxxBlasTest, dotTest) {
  const int n = 1000;
  ::std::mt19937 gen;
  ::std::uniform_real_distribution< double > dist;

  float sx[n], sy[n];
  float sr = 0.0, sh = 0.0;
  for (int i = 0; i < n; ++i) {
    sx[i] = static_cast< float >(dist(gen));
    sy[i] = static_cast< float >(dist(gen));
    sr += sx[i] * sy[i];
    if (i % 2 == 0) {
      sh += sx[i] * sy[i];
    }
  }
  EXPECT_FLOAT_EQ(sr, cxxblas::dot(n, sx, sy));
  EXPECT_FLOAT_EQ(sh, cxxblas::dot(n / 2, sx, sy, 2, 2));

  double dx[n], dy[n];
  double dr = 0.0, dh = 0.0;
  for (int i = 0; i < n; ++i) {
    dx[i] = static_cast< double >(dist(gen));
    dy[i] = static_cast< double >(dist(gen));
    dr += dx[i] * dy[i];
    if (i % 2 == 0) {
      dh += dx[i] * dy[i];
    }
  }
  EXPECT_FLOAT_EQ(dr, cxxblas::dot(n, dx, dy));
  EXPECT_FLOAT_EQ(dh, cxxblas::dot(n / 2, dx, dy, 2, 2));

  ::std::complex< float > cx[n], cy[n];
  ::std::complex< float > cr = 0.0, ch = 0.0;
  for (int i = 0; i < n; ++i) {
    cx[i] = ::std::complex< float >(dist(gen), dist(gen));
    cy[i] = ::std::complex< float >(dist(gen), dist(gen));
    cr += cx[i] * cy[i];
    if (i % 2 == 0) {
      ch += cx[i] * cy[i];
    }
  }
  EXPECT_FLOAT_EQ(::std::real(cr), ::std::real(cxxblas::dot(n, cx, cy)));
  EXPECT_FLOAT_EQ(::std::imag(cr), ::std::imag(cxxblas::dot(n, cx, cy)));
  EXPECT_FLOAT_EQ(::std::real(ch), ::std::real(
      cxxblas::dot(n / 2, cx, cy, 2, 2)));
  EXPECT_FLOAT_EQ(::std::imag(ch), ::std::imag(
      cxxblas::dot(n / 2, cx, cy, 2, 2)));

  ::std::complex< double > zx[n], zy[n];
  ::std::complex< double > zr = 0.0, zh = 0.0;
  for (int i = 0; i < n; ++i) {
    zx[i] = ::std::complex< double >(dist(gen), dist(gen));
    zy[i] = ::std::complex< double >(dist(gen), dist(gen));
    zr += zx[i] * zy[i];
    if (i % 2 == 0) {
      zh += zx[i] * zy[i];
    }
  }
  EXPECT_FLOAT_EQ(::std::real(zr), ::std::real(cxxblas::dot(n, zx, zy)));
  EXPECT_FLOAT_EQ(::std::imag(zr), ::std::imag(cxxblas::dot(n, zx, zy)));
  EXPECT_FLOAT_EQ(::std::real(zh), ::std::real(
      cxxblas::dot(n / 2, zx, zy, 2, 2)));
  EXPECT_FLOAT_EQ(::std::imag(zh), ::std::imag(
      cxxblas::dot(n / 2, zx, zy, 2, 2)));
}

TEST(CxxBlasTest, dotcTest) {
  const int n = 1000;
  ::std::mt19937 gen;
  ::std::uniform_real_distribution< double > dist;

  float sx[n], sy[n];
  float sr = 0.0, sh = 0.0;
  for (int i = 0; i < n; ++i) {
    sx[i] = static_cast< float >(dist(gen));
    sy[i] = static_cast< float >(dist(gen));
    sr += sx[i] * sy[i];
    if (i % 2 == 0) {
      sh += sx[i] * sy[i];
    }
  }
  EXPECT_FLOAT_EQ(sr, cxxblas::dotc(n, sx, sy));
  EXPECT_FLOAT_EQ(sh, cxxblas::dotc(n / 2, sx, sy, 2, 2));

  double dx[n], dy[n];
  double dr = 0.0, dh = 0.0;
  for (int i = 0; i < n; ++i) {
    dx[i] = static_cast< double >(dist(gen));
    dy[i] = static_cast< double >(dist(gen));
    dr += dx[i] * dy[i];
    if (i % 2 == 0) {
      dh += dx[i] * dy[i];
    }
  }
  EXPECT_FLOAT_EQ(dr, cxxblas::dotc(n, dx, dy));
  EXPECT_FLOAT_EQ(dh, cxxblas::dotc(n / 2, dx, dy, 2, 2));

  ::std::complex< float > cx[n], cy[n];
  ::std::complex< float > cr = 0.0, ch = 0.0;
  for (int i = 0; i < n; ++i) {
    cx[i] = ::std::complex< float >(dist(gen), dist(gen));
    cy[i] = ::std::complex< float >(dist(gen), dist(gen));
    cr += ::std::conj(cx[i]) * cy[i];
    if (i % 2 == 0) {
      ch += ::std::conj(cx[i]) * cy[i];
    }
  }
  EXPECT_FLOAT_EQ(::std::real(cr), ::std::real(cxxblas::dotc(n, cx, cy)));
  EXPECT_FLOAT_EQ(::std::imag(cr), ::std::imag(cxxblas::dotc(n, cx, cy)));
  EXPECT_FLOAT_EQ(::std::real(ch), ::std::real(
      cxxblas::dotc(n / 2, cx, cy, 2, 2)));
  EXPECT_FLOAT_EQ(::std::imag(ch), ::std::imag(
      cxxblas::dotc(n / 2, cx, cy, 2, 2)));

  ::std::complex< double > zx[n], zy[n];
  ::std::complex< double > zr = 0.0, zh = 0.0;
  for (int i = 0; i < n; ++i) {
    zx[i] = ::std::complex< double >(dist(gen), dist(gen));
    zy[i] = ::std::complex< double >(dist(gen), dist(gen));
    zr += ::std::conj(zx[i]) * zy[i];
    if (i % 2 == 0) {
      zh += ::std::conj(zx[i]) * zy[i];
    }
  }
  EXPECT_FLOAT_EQ(::std::real(zr), ::std::real(cxxblas::dotc(n, zx, zy)));
  EXPECT_FLOAT_EQ(::std::imag(zr), ::std::imag(cxxblas::dotc(n, zx, zy)));
  EXPECT_FLOAT_EQ(::std::real(zh), ::std::real(
      cxxblas::dotc(n / 2, zx, zy, 2, 2)));
  EXPECT_FLOAT_EQ(::std::imag(zh), ::std::imag(
      cxxblas::dotc(n / 2, zx, zy, 2, 2)));
}

TEST(CxxBlasTest, dotuTest) {
  const int n = 1000;
  ::std::mt19937 gen;
  ::std::uniform_real_distribution< double > dist;

  float sx[n], sy[n];
  float sr = 0.0, sh = 0.0;
  for (int i = 0; i < n; ++i) {
    sx[i] = static_cast< float >(dist(gen));
    sy[i] = static_cast< float >(dist(gen));
    sr += sx[i] * sy[i];
    if (i % 2 == 0) {
      sh += sx[i] * sy[i];
    }
  }
  EXPECT_FLOAT_EQ(sr, cxxblas::dotu(n, sx, sy));
  EXPECT_FLOAT_EQ(sh, cxxblas::dotu(n / 2, sx, sy, 2, 2));

  double dx[n], dy[n];
  double dr = 0.0, dh = 0.0;
  for (int i = 0; i < n; ++i) {
    dx[i] = static_cast< double >(dist(gen));
    dy[i] = static_cast< double >(dist(gen));
    dr += dx[i] * dy[i];
    if (i % 2 == 0) {
      dh += dx[i] * dy[i];
    }
  }
  EXPECT_FLOAT_EQ(dr, cxxblas::dotu(n, dx, dy));
  EXPECT_FLOAT_EQ(dh, cxxblas::dotu(n / 2, dx, dy, 2, 2));

  ::std::complex< float > cx[n], cy[n];
  ::std::complex< float > cr = 0.0, ch = 0.0;
  for (int i = 0; i < n; ++i) {
    cx[i] = ::std::complex< float >(dist(gen), dist(gen));
    cy[i] = ::std::complex< float >(dist(gen), dist(gen));
    cr += cx[i] * cy[i];
    if (i % 2 == 0) {
      ch += cx[i] * cy[i];
    }
  }
  EXPECT_FLOAT_EQ(::std::real(cr), ::std::real(cxxblas::dotu(n, cx, cy)));
  EXPECT_FLOAT_EQ(::std::imag(cr), ::std::imag(cxxblas::dotu(n, cx, cy)));
  EXPECT_FLOAT_EQ(::std::real(ch), ::std::real(
      cxxblas::dotu(n / 2, cx, cy, 2, 2)));
  EXPECT_FLOAT_EQ(::std::imag(ch), ::std::imag(
      cxxblas::dotu(n / 2, cx, cy, 2, 2)));

  ::std::complex< double > zx[n], zy[n];
  ::std::complex< double > zr = 0.0, zh = 0.0;
  for (int i = 0; i < n; ++i) {
    zx[i] = ::std::complex< double >(dist(gen), dist(gen));
    zy[i] = ::std::complex< double >(dist(gen), dist(gen));
    zr += zx[i] * zy[i];
    if (i % 2 == 0) {
      zh += zx[i] * zy[i];
    }
  }
  EXPECT_FLOAT_EQ(::std::real(zr), ::std::real(cxxblas::dotu(n, zx, zy)));
  EXPECT_FLOAT_EQ(::std::imag(zr), ::std::imag(cxxblas::dotu(n, zx, zy)));
  EXPECT_FLOAT_EQ(::std::real(zh), ::std::real(
      cxxblas::dotu(n / 2, zx, zy, 2, 2)));
  EXPECT_FLOAT_EQ(::std::imag(zh), ::std::imag(
      cxxblas::dotu(n / 2, zx, zy, 2, 2)));
}

TEST(CxxBlasTest, nrm2Test) {
  const int n = 1000;
  ::std::mt19937 gen;
  ::std::uniform_real_distribution< double > dist;

  float sx[n], ss = 0.0, sh = 0.0;
  for (int i = 0; i < n; ++i) {
    sx[i] = static_cast< float >(dist(gen));
    ss += sx[i] * sx[i];
    if (i % 2 == 0) {
      sh += sx[i] * sx[i];
    }
  }
  ss = ::std::sqrt(ss);
  sh = ::std::sqrt(sh);
  EXPECT_FLOAT_EQ(ss, cxxblas::nrm2(n, sx));
  EXPECT_FLOAT_EQ(sh, cxxblas::nrm2(n / 2, sx, 2));

  double dx[n], ds = 0.0, dh = 0.0;
  for (int i = 0; i < n; ++i) {
    dx[i] = static_cast< double >(dist(gen));
    ds += dx[i] * dx[i];
    if (i % 2 == 0) {
      dh += dx[i] * dx[i];
    }
  }
  ds = ::std::sqrt(ds);
  dh = ::std::sqrt(dh);
  EXPECT_FLOAT_EQ(ds, cxxblas::nrm2(n, dx));
  EXPECT_FLOAT_EQ(dh, cxxblas::nrm2(n / 2, dx, 2));

  ::std::complex< float > cx[n];
  float cs = 0.0, ch = 0.0;
  for (int i = 0; i < n; ++i) {
    cx[i] = ::std::complex< float >(dist(gen), dist(gen));
    cs += ::std::norm(cx[i]);
    if (i % 2 == 0) {
      ch += ::std::norm(cx[i]);
    }
  }
  cs = ::std::sqrt(cs);
  ch = ::std::sqrt(ch);
  EXPECT_FLOAT_EQ(cs, cxxblas::nrm2(n, cx));
  EXPECT_FLOAT_EQ(ch, cxxblas::nrm2(n / 2, cx, 2));

  ::std::complex< double > zx[n];
  double zs = 0.0, zh = 0.0;
  for (int i = 0; i < n; ++i) {
    zx[i] = ::std::complex< double >(dist(gen), dist(gen));
    zs += ::std::norm(zx[i]);
    if (i % 2 == 0) {
      zh += ::std::norm(zx[i]);
    }
  }
  zs = ::std::sqrt(zs);
  zh = ::std::sqrt(zh);
  EXPECT_FLOAT_EQ(zs, cxxblas::nrm2(n, zx));
  EXPECT_FLOAT_EQ(zh, cxxblas::nrm2(n / 2, zx, 2));
}

TEST(CxxBlasTest, rotTest) {
  const int n = 1000;
  ::std::mt19937 gen;
  ::std::uniform_real_distribution< double > dist;

  float sx[n], sy[n], su[n], sv[n], sp[n], sq[n];
  float sc = static_cast< float >(dist(gen));
  float ss = static_cast< float >(dist(gen));
  for (int i = 0; i < n; ++i) {
    sx[i] = static_cast< float >(dist(gen));
    sy[i] = static_cast< float >(dist(gen));
    su[i] = sx[i];
    sv[i] = sy[i];
    if (i % 2 == 0) {
      sp[i] = sx[i];
      sq[i] = sy[i];
    } else {
      sp[i] = static_cast< float >(dist(gen));
      sq[i] = static_cast< float >(dist(gen));
    }
  }
  cxxblas::rot(n, su, sv);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(sx[i] + sy[i], su[i]);
    EXPECT_FLOAT_EQ(sy[i] - sx[i], sv[i]);
  }
  cxxblas::rot(n / 2, sp, sq, sc, ss, 2, 2);
  for (int i = 0; i < n / 2; ++i) {
    EXPECT_FLOAT_EQ(sc * sx[2 * i] + ss * sy[2 * i], sp[2 * i]);
    EXPECT_FLOAT_EQ(sc * sy[2 * i] - ss * sx[2 * i], sq[2 * i]);
  }

  double dx[n], dy[n], du[n], dv[n], dp[n], dq[n];
  double dc = static_cast< double >(dist(gen));
  double ds = static_cast< double >(dist(gen));
  for (int i = 0; i < n; ++i) {
    dx[i] = static_cast< double >(dist(gen));
    dy[i] = static_cast< double >(dist(gen));
    du[i] = dx[i];
    dv[i] = dy[i];
    if (i % 2 == 0) {
      dp[i] = dx[i];
      dq[i] = dy[i];
    } else {
      dp[i] = static_cast< double >(dist(gen));
      dq[i] = static_cast< double >(dist(gen));
    }
  }
  cxxblas::rot(n, du, dv);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(dx[i] + dy[i], du[i]);
    EXPECT_FLOAT_EQ(dy[i] - dx[i], dv[i]);
  }
  cxxblas::rot(n / 2, dp, dq, dc, ds, 2, 2);
  for (int i = 0; i < n / 2; ++i) {
    EXPECT_FLOAT_EQ(dc * dx[2 * i] + ds * dy[2 * i], dp[2 * i]);
    EXPECT_FLOAT_EQ(dc * dy[2 * i] - ds * dx[2 * i], dq[2 * i]);
  }

  ::std::complex< float > cx[n], cy[n], cu[n], cv[n], cp[n], cq[n];
  float cc = static_cast< float >(dist(gen));
  float cs = static_cast< float >(dist(gen));
  for (int i = 0; i < n; ++i) {
    cx[i] = ::std::complex< float >(dist(gen), dist(gen));
    cy[i] = ::std::complex< float >(dist(gen), dist(gen));
    cu[i] = cx[i];
    cv[i] = cy[i];
    if (i % 2 == 0) {
      cp[i] = cx[i];
      cq[i] = cy[i];
    } else {
      cp[i] = ::std::complex< float >(dist(gen), dist(gen));
      cq[i] = ::std::complex< float >(dist(gen), dist(gen));
    }
  }
  cxxblas::rot(n, cu, cv);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(::std::real(cx[i] + cy[i]), ::std::real(cu[i]));
    EXPECT_FLOAT_EQ(::std::imag(cx[i] + cy[i]), ::std::imag(cu[i]));
    EXPECT_FLOAT_EQ(::std::real(cy[i] - cx[i]), ::std::real(cv[i]));
    EXPECT_FLOAT_EQ(::std::imag(cy[i] - cx[i]), ::std::imag(cv[i]));
  }
  cxxblas::rot(n / 2, cp, cq, cc, cs, 2, 2);
  for (int i = 0; i < n / 2; ++i) {
    EXPECT_FLOAT_EQ(
        ::std::real(::std::complex< float >(cc) * cx[2 * i] +
                    ::std::complex< float >(cs) * cy[2 * i]),
        ::std::real(cp[2 * i]));
    EXPECT_FLOAT_EQ(
        ::std::imag(::std::complex< float >(cc) * cx[2 * i] +
                    ::std::complex< float >(cs) * cy[2 * i]),
        ::std::imag(cp[2 * i]));
    EXPECT_FLOAT_EQ(
        ::std::real(::std::complex< float >(cc) * cy[2 * i] -
                    ::std::complex< float >(cs) * cx[2 * i]),
        ::std::real(cq[2 * i]));
    EXPECT_FLOAT_EQ(
        ::std::imag(::std::complex< float >(cc) * cy[2 * i] -
                    ::std::complex< float >(cs) * cx[2 * i]),
        ::std::imag(cq[2 * i]));
  }

  ::std::complex< double > zx[n], zy[n], zu[n], zv[n], zp[n], zq[n];
  double zc = static_cast< double >(dist(gen));
  double zs = static_cast< double >(dist(gen));
  for (int i = 0; i < n; ++i) {
    zx[i] = ::std::complex< double >(dist(gen), dist(gen));
    zy[i] = ::std::complex< double >(dist(gen), dist(gen));
    zu[i] = zx[i];
    zv[i] = zy[i];
    if (i % 2 == 0) {
      zp[i] = zx[i];
      zq[i] = zy[i];
    } else {
      zp[i] = ::std::complex< double >(dist(gen), dist(gen));
      zq[i] = ::std::complex< double >(dist(gen), dist(gen));
    }
  }
  cxxblas::rot(n, zu, zv);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(::std::real(zx[i] + zy[i]), ::std::real(zu[i]));
    EXPECT_FLOAT_EQ(::std::imag(zx[i] + zy[i]), ::std::imag(zu[i]));
    EXPECT_FLOAT_EQ(::std::real(zy[i] - zx[i]), ::std::real(zv[i]));
    EXPECT_FLOAT_EQ(::std::imag(zy[i] - zx[i]), ::std::imag(zv[i]));
  }
  cxxblas::rot(n / 2, zp, zq, zc, zs, 2, 2);
  for (int i = 0; i < n / 2; ++i) {
    EXPECT_FLOAT_EQ(
        ::std::real(::std::complex< double >(zc) * zx[2 * i] +
                    ::std::complex< double >(zs) * zy[2 * i]),
        ::std::real(zp[2 * i]));
    EXPECT_FLOAT_EQ(
        ::std::imag(::std::complex< double >(zc) * zx[2 * i] +
                    ::std::complex< double >(zs) * zy[2 * i]),
        ::std::imag(zp[2 * i]));
    EXPECT_FLOAT_EQ(
        ::std::real(::std::complex< double >(zc) * zy[2 * i] -
                    ::std::complex< double >(zs) * zx[2 * i]),
        ::std::real(zq[2 * i]));
    EXPECT_FLOAT_EQ(
        ::std::imag(::std::complex< double >(zc) * zy[2 * i] -
                    ::std::complex< double >(zs) * zx[2 * i]),
        ::std::imag(zq[2 * i]));
  }
}

TEST(CxxBlasTest, rotgTest) {
  ::std::mt19937 gen;
  ::std::uniform_real_distribution< double > dist;

  float sa = static_cast< float >(dist(gen));
  float sb = static_cast< float >(dist(gen));
  float sr = sa, sz = sb, sc = 0.0, ss = 0.0;
  cxxblas::rotg(&sr, &sz, &sc, &ss);
  EXPECT_FLOAT_EQ(sc * sa + ss * sb, sr);
  EXPECT_FLOAT_EQ(sc * sb - ss * sa, 0);
  if (::std::abs(sa) > ::std::abs(sb)) {
    EXPECT_FLOAT_EQ(ss, sz);
  }  else if (sc != 0) {
    EXPECT_FLOAT_EQ(1.0 / sc, sz);
  } else {
    EXPECT_FLOAT_EQ(1.0, sz);
  }

  double da = static_cast< double >(dist(gen));
  double db = static_cast< double >(dist(gen));
  double dr = da, dz = db, dc = 0.0, ds = 0.0;
  cxxblas::rotg(&dr, &dz, &dc, &ds);
  EXPECT_FLOAT_EQ(dc * da + ds * db, dr);
  EXPECT_FLOAT_EQ(dc * db - ds * da, 0);
  if (::std::abs(da) > ::std::abs(db)) {
    EXPECT_FLOAT_EQ(ds, dz);
  }  else if (dc != 0) {
    EXPECT_FLOAT_EQ(1.0 / dc, dz);
  } else {
    EXPECT_FLOAT_EQ(1.0, dz);
  }

  ::std::complex< float > ca = ::std::complex< float >(dist(gen), dist(gen));
  ::std::complex< float > cb = ::std::complex< float >(dist(gen), dist(gen));
  ::std::complex< float > cr = ca, cz = cb, cs = 0.0;
  float cc = 0.0;
  cxxblas::rotg(&cr, &cz, &cc, &cs);
  EXPECT_FLOAT_EQ(::std::real(::std::complex< float >(cc) * ca + cs * cb),
                  ::std::real(cr));
  EXPECT_FLOAT_EQ(::std::imag(::std::complex< float >(cc) * ca + cs * cb),
                  ::std::imag(cr));

  ::std::complex< double > za = ::std::complex< double >(dist(gen), dist(gen));
  ::std::complex< double > zb = ::std::complex< double >(dist(gen), dist(gen));
  ::std::complex< double > zr = za, zz = zb, zs = 0.0;
  double zc = 0.0;
  cxxblas::rotg(&zr, &zz, &zc, &zs);
  EXPECT_FLOAT_EQ(::std::real(::std::complex< double >(zc) * za + zs * zb),
                  ::std::real(zr));
  EXPECT_FLOAT_EQ(::std::imag(::std::complex< double >(zc) * za + zs * zb),
                  ::std::imag(zr));
}

TEST(CxxBlasTest, rotmTest) {
  const int n = 1000;
  ::std::mt19937 gen;
  ::std::uniform_real_distribution< double > dist;

  float sx[n], sy[n], su[n], sv[n], sp[n], sq[n];
  float sa[5] = {-1.0, static_cast< float >(dist(gen)),
                 static_cast< float >(dist(gen)),
                 static_cast< float >(dist(gen)),
                 static_cast< float >(dist(gen))};
  for (int i = 0; i < n; ++i) {
    sx[i] = static_cast< float >(dist(gen));
    sy[i] = static_cast< float >(dist(gen));
    su[i] = sx[i];
    sv[i] = sy[i];
    if (i % 2 == 0) {
      sp[i] = sx[i];
      sq[i] = sy[i];
    } else {
      sp[i] = static_cast< float >(dist(gen));
      sq[i] = static_cast< float >(dist(gen));
    }
  }
  cxxblas::rotm(n, su, sv, sa);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(sa[1] * sx[i] + sa[3] * sy[i], su[i]);
    EXPECT_FLOAT_EQ(sa[2] * sx[i] + sa[4] * sy[i], sv[i]);
  }
  cxxblas::rotm(n / 2, sp, sq, sa, 2, 2);
  for (int i = 0; i < n / 2; ++i) {
    EXPECT_FLOAT_EQ(sa[1] * sx[2 * i] + sa[3] * sy[2 * i], sp[2 * i]);
    EXPECT_FLOAT_EQ(sa[2] * sx[2 * i] + sa[4] * sy[2 * i], sq[2 * i]);
  }

  double dx[n], dy[n], du[n], dv[n], dp[n], dq[n];
  double da[5] = {-1.0, static_cast< double >(dist(gen)),
                  static_cast< double >(dist(gen)),
                  static_cast< double >(dist(gen)),
                  static_cast< double >(dist(gen))};
  for (int i = 0; i < n; ++i) {
    dx[i] = static_cast< double >(dist(gen));
    dy[i] = static_cast< double >(dist(gen));
    du[i] = dx[i];
    dv[i] = dy[i];
    if (i % 2 == 0) {
      dp[i] = dx[i];
      dq[i] = dy[i];
    } else {
      dp[i] = static_cast< double >(dist(gen));
      dq[i] = static_cast< double >(dist(gen));
    }
  }
  cxxblas::rotm(n, du, dv, da);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(da[1] * dx[i] + da[3] * dy[i], du[i]);
    EXPECT_FLOAT_EQ(da[2] * dx[i] + da[4] * dy[i], dv[i]);
  }
  cxxblas::rotm(n / 2, dp, dq, da, 2, 2);
  for (int i = 0; i < n / 2; ++i) {
    EXPECT_FLOAT_EQ(da[1] * dx[2 * i] + da[3] * dy[2 * i], dp[2 * i]);
    EXPECT_FLOAT_EQ(da[2] * dx[2 * i] + da[4] * dy[2 * i], dq[2 * i]);
  }
}

// TODO: Figure out what rotmg is doing and refine tests
TEST(CxxBlasTest, rotmgTest) {
  ::std::mt19937 gen;
  ::std::uniform_real_distribution< double > dist;

  float s1 = static_cast< float >(dist(gen));
  float s2 = static_cast< float >(dist(gen));
  float sx = static_cast< float >(dist(gen));
  float sy = static_cast< float >(dist(gen));
  float sa[5];
  cxxblas::rotmg(&s1, &s2, &sx, sy, sa);

  double d1 = static_cast< double >(dist(gen));
  double d2 = static_cast< double >(dist(gen));
  double dx = static_cast< double >(dist(gen));
  double dy = static_cast< double >(dist(gen));
  double da[5];
  cxxblas::rotmg(&d1, &d2, &dx, dy, da);
}

TEST(CxxBlasTest, scalTest) {
  const int n = 1000;
  ::std::mt19937 gen;
  ::std::uniform_real_distribution< double > dist;

  float sx[n], sr[n], sh[n];
  float sa = static_cast< float >(dist(gen));
  for (int i = 0; i < n; ++i) {
    sx[i] = static_cast< float >(dist(gen));
    sr[i] = sx[i];
    if (i % 2 == 0) {
      sh[i] = sx[i];
    } else {
      sh[i] = static_cast< float >(dist(gen));
    }
  }
  cxxblas::scal(n, sr, sa);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(sa * sx[i], sr[i]);
  }
  cxxblas::scal(n / 2, sh, sa, 2);
  for (int i = 0; i < n / 2; ++i) {
    EXPECT_FLOAT_EQ(sa * sx[i * 2], sh[i * 2]);
  }

  double dx[n], dr[n], dh[n];
  double da = static_cast< double >(dist(gen));
  for (int i = 0; i < n; ++i) {
    dx[i] = static_cast< double >(dist(gen));
    dr[i] = dx[i];
    if (i % 2 == 0) {
      dh[i] = dx[i];
    } else {
      dh[i] = static_cast< double >(dist(gen));
    }
  }
  cxxblas::scal(n, dr, da);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(da * dx[i], dr[i]);
  }
  cxxblas::scal(n / 2, dh, da, 2);
  for (int i = 0; i < n / 2; ++i) {
    EXPECT_FLOAT_EQ(da * dx[i * 2], dh[i * 2]);
  }

  ::std::complex< float > cx[n], cr[n], ch[n];
  ::std::complex< float > ca = ::std::complex< float >(dist(gen), dist(gen));
  for (int i = 0; i < n; ++i) {
    cx[i] = ::std::complex< float >(dist(gen), dist(gen));
    cr[i] = cx[i];
    if (i % 2 == 0) {
      ch[i] = cx[i];
    } else {
      ch[i] = ::std::complex< float >(dist(gen), dist(gen));
    }
  }
  cxxblas::scal(n, cr, ca);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(::std::real(ca * cx[i]), ::std::real(cr[i]));
    EXPECT_FLOAT_EQ(::std::imag(ca * cx[i]), ::std::imag(cr[i]));
  }
  cxxblas::scal(n / 2, ch, ca, 2);
  for (int i = 0; i < n / 2; ++i) {
    EXPECT_FLOAT_EQ(::std::real(ca * cx[i * 2]), ::std::real(ch[i * 2]));
    EXPECT_FLOAT_EQ(::std::imag(ca * cx[i * 2]), ::std::imag(ch[i * 2]));
  }

  ::std::complex< double > zx[n], zr[n], zh[n];
  ::std::complex< double > za = ::std::complex< double >(dist(gen), dist(gen));
  for (int i = 0; i < n; ++i) {
    zx[i] = ::std::complex< double >(dist(gen), dist(gen));
    zr[i] = zx[i];
    if (i % 2 == 0) {
      zh[i] = zx[i];
    } else {
      zh[i] = ::std::complex< double >(dist(gen), dist(gen));
    }
  }
  cxxblas::scal(n, zr, za);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(::std::real(za * zx[i]), ::std::real(zr[i]));
    EXPECT_FLOAT_EQ(::std::imag(za * zx[i]), ::std::imag(zr[i]));
  }
  cxxblas::scal(n / 2, zh, za, 2);
  for (int i = 0; i < n / 2; ++i) {
    EXPECT_FLOAT_EQ(::std::real(za * zx[i * 2]), ::std::real(zh[i * 2]));
    EXPECT_FLOAT_EQ(::std::imag(za * zx[i * 2]), ::std::imag(zh[i * 2]));
  }

  ::std::complex< float > csx[n], csr[n], csh[n];
  float csa = static_cast< float >(dist(gen));
  for (int i = 0; i < n; ++i) {
    csx[i] = ::std::complex< float >(dist(gen), dist(gen));
    csr[i] = csx[i];
    if (i % 2 == 0) {
      csh[i] = csx[i];
    } else {
      csh[i] = ::std::complex< float >(dist(gen), dist(gen));
    }
  }
  cxxblas::scal(n, csr, csa);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(::std::real(::std::complex< float >(csa) * csx[i]),
                    ::std::real(csr[i]));
    EXPECT_FLOAT_EQ(::std::imag(::std::complex< float >(csa) * csx[i]),
                    ::std::imag(csr[i]));
  }
  cxxblas::scal(n / 2, csh, csa, 2);
  for (int i = 0; i < n / 2; ++i) {
    EXPECT_FLOAT_EQ(::std::real(::std::complex< float >(csa) * csx[i * 2]),
                    ::std::real(csh[i * 2]));
    EXPECT_FLOAT_EQ(::std::imag(::std::complex< float >(csa) * csx[i * 2]),
                    ::std::imag(csh[i * 2]));
  }

  ::std::complex< double > zdx[n], zdr[n], zdh[n];
  double zda = static_cast< double >(dist(gen));
  for (int i = 0; i < n; ++i) {
    zdx[i] = ::std::complex< double >(dist(gen), dist(gen));
    zdr[i] = zdx[i];
    if (i % 2 == 0) {
      zdh[i] = zdx[i];
    } else {
      zdh[i] = ::std::complex< double >(dist(gen), dist(gen));
    }
  }
  cxxblas::scal(n, zdr, zda);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(::std::real(::std::complex< double >(zda) * zdx[i]),
                    ::std::real(zdr[i]));
    EXPECT_FLOAT_EQ(::std::imag(::std::complex< double >(zda) * zdx[i]),
                    ::std::imag(zdr[i]));
  }
  cxxblas::scal(n / 2, zdh, zda, 2);
  for (int i = 0; i < n / 2; ++i) {
    EXPECT_FLOAT_EQ(::std::real(::std::complex< double >(zda) * zdx[i * 2]),
                    ::std::real(zdh[i * 2]));
    EXPECT_FLOAT_EQ(::std::imag(::std::complex< double >(zda) * zdx[i * 2]),
                    ::std::imag(zdh[i * 2]));
  }
}

TEST(CxxBlasTest, swapTest) {
  const int n = 1000;
  ::std::mt19937 gen;
  ::std::uniform_real_distribution< double > dist;

  float sx[n], sy[n], su[n], sv[n], sp[n], sq[n];
  for (int i = 0; i < n; ++i) {
    sx[i] = static_cast< float >(dist(gen));
    sy[i] = static_cast< float >(dist(gen));
    su[i] = sx[i];
    sv[i] = sy[i];
    if (i % 2 == 0) {
      sp[i] = sx[i];
      sq[i] = sy[i];
    } else {
      sp[i] = static_cast< float >(dist(gen));
      sq[i] = static_cast< float >(dist(gen));
    }
  }
  cxxblas::swap(n, su, sv);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(sx[i], sv[i]);
    EXPECT_FLOAT_EQ(sy[i], su[i]);
  }
  cxxblas::swap(n / 2, sp, sq, 2, 2);
  for (int i = 0; i < n / 2; ++i) {
    EXPECT_FLOAT_EQ(sx[i * 2], sq[i * 2]);
    EXPECT_FLOAT_EQ(sy[i * 2], sp[i * 2]);
  }

  double dx[n], dy[n], du[n], dv[n], dp[n], dq[n];
  for (int i = 0; i < n; ++i) {
    dx[i] = static_cast< double >(dist(gen));
    dy[i] = static_cast< double >(dist(gen));
    du[i] = dx[i];
    dv[i] = dy[i];
    if (i % 2 == 0) {
      dp[i] = dx[i];
      dq[i] = dy[i];
    } else {
      dp[i] = static_cast< double >(dist(gen));
      dq[i] = static_cast< double >(dist(gen));
    }
  }
  cxxblas::swap(n, du, dv);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(dx[i], dv[i]);
    EXPECT_FLOAT_EQ(dy[i], du[i]);
  }
  cxxblas::swap(n / 2, dp, dq, 2, 2);
  for (int i = 0; i < n / 2; ++i) {
    EXPECT_FLOAT_EQ(dx[i * 2], dq[i * 2]);
    EXPECT_FLOAT_EQ(dy[i * 2], dp[i * 2]);
  }

  ::std::complex< float > cx[n], cy[n], cu[n], cv[n], cp[n], cq[n];
  for (int i = 0; i < n; ++i) {
    cx[i] = ::std::complex< float >(dist(gen), dist(gen));
    cy[i] = ::std::complex< float >(dist(gen), dist(gen));
    cu[i] = cx[i];
    cv[i] = cy[i];
    if (i % 2 == 0) {
      cp[i] = cx[i];
      cq[i] = cy[i];
    } else {
      cp[i] = ::std::complex< float >(dist(gen), dist(gen));
      cq[i] = ::std::complex< float >(dist(gen), dist(gen));
    }
  }
  cxxblas::swap(n, cu, cv);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(::std::real(cx[i]), ::std::real(cv[i]));
    EXPECT_FLOAT_EQ(::std::imag(cx[i]), ::std::imag(cv[i]));
    EXPECT_FLOAT_EQ(::std::real(cy[i]), ::std::real(cu[i]));
    EXPECT_FLOAT_EQ(::std::imag(cy[i]), ::std::imag(cu[i]));
  }
  cxxblas::swap(n / 2, cp, cq, 2, 2);
  for (int i = 0; i < n / 2; ++i) {
    EXPECT_FLOAT_EQ(::std::real(cx[i * 2]), ::std::real(cq[i * 2]));
    EXPECT_FLOAT_EQ(::std::imag(cx[i * 2]), ::std::imag(cq[i * 2]));
    EXPECT_FLOAT_EQ(::std::real(cy[i * 2]), ::std::real(cp[i * 2]));
    EXPECT_FLOAT_EQ(::std::imag(cy[i * 2]), ::std::imag(cp[i * 2]));
  }

  ::std::complex< double > zx[n], zy[n], zu[n], zv[n], zp[n], zq[n];
  for (int i = 0; i < n; ++i) {
    zx[i] = ::std::complex< double >(dist(gen), dist(gen));
    zy[i] = ::std::complex< double >(dist(gen), dist(gen));
    zu[i] = zx[i];
    zv[i] = zy[i];
    if (i % 2 == 0) {
      zp[i] = zx[i];
      zq[i] = zy[i];
    } else {
      zp[i] = ::std::complex< double >(dist(gen), dist(gen));
      zq[i] = ::std::complex< double >(dist(gen), dist(gen));
    }
  }
  cxxblas::swap(n, zu, zv);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(::std::real(zx[i]), ::std::real(zv[i]));
    EXPECT_FLOAT_EQ(::std::imag(zx[i]), ::std::imag(zv[i]));
    EXPECT_FLOAT_EQ(::std::real(zy[i]), ::std::real(zu[i]));
    EXPECT_FLOAT_EQ(::std::imag(zy[i]), ::std::imag(zu[i]));
  }
  cxxblas::swap(n / 2, zp, zq, 2, 2);
  for (int i = 0; i < n / 2; ++i) {
    EXPECT_FLOAT_EQ(::std::real(zx[i * 2]), ::std::real(zq[i * 2]));
    EXPECT_FLOAT_EQ(::std::imag(zx[i * 2]), ::std::imag(zq[i * 2]));
    EXPECT_FLOAT_EQ(::std::real(zy[i * 2]), ::std::real(zp[i * 2]));
    EXPECT_FLOAT_EQ(::std::imag(zy[i * 2]), ::std::imag(zp[i * 2]));
  }
}

// TODO: Figure out why it returns wrong indices
TEST(CxxBlasTest, iamaxTest) {
  const int n = 1000;
  ::std::mt19937 gen;
  ::std::uniform_real_distribution< double > dist;

  float sx[n], sp[n];
  for (int i = 0; i < n; ++i) {
    sx[i] = static_cast< float >(dist(gen));
    if (i % 2 == 0) {
      sp[i] = sx[i];
    } else {
      sp[i] = static_cast< float >(dist(gen));
    }
  }
  cxxblas::iamax(n, sx);
  cxxblas::iamax(n / 2, sp, 2);

  double dx[n], dp[n];
  for (int i = 0; i < n; ++i) {
    dx[i] = static_cast< double >(dist(gen));
    if (i % 2 == 0) {
      dp[i] = dx[i];
    } else {
      dp[i] = static_cast< double >(dist(gen));
    }
  }
  cxxblas::iamax(n, dx);
  cxxblas::iamax(n / 2, dp, 2);

  ::std::complex< float > cx[n], cp[n];
  for (int i = 0; i < n; ++i) {
    cx[i] = ::std::complex< float >(dist(gen), dist(gen));
    if (i % 2 == 0) {
      cp[i] = cx[i];
    } else {
      cp[i] = ::std::complex< float >(dist(gen), dist(gen));
    }
  }
  cxxblas::iamax(n, cx);
  cxxblas::iamax(n / 2, cp, 2);


  ::std::complex< double > zx[n], zp[n];
  for (int i = 0; i < n; ++i) {
    zx[i] = ::std::complex< double >(dist(gen), dist(gen));
    if (i % 2 == 0) {
      zp[i] = zx[i];
    } else {
      zp[i] = ::std::complex< double >(dist(gen), dist(gen));
    }
  }
  cxxblas::iamax(n, zx);
  cxxblas::iamax(n / 2, zp, 2);
}

}  // namespace
}  // namespace linalg
}  // namespace thunder
