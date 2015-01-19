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

#include "thunder/linalg/cxxblas.hpp"

#include <complex>

#include "thunder/linalg/blas.hpp"

namespace thunder {
namespace linalg {
namespace cxxblas {

float asum(int n, const float *x, int incx) { return sasum_(&n, x, &incx); }

double asum(int n, const double *x, int incx) { return dasum_(&n, x, &incx); }

float asum(int n, const ::std::complex< float > *x, int incx) {
  return scasum_(&n, x, &incx);
}

double asum(int n, const ::std::complex< double > *x, int incx) {
  return dzasum_(&n, x, &incx);
}

void axpy(int n, const float *x, float *y, float a, int incx, int incy) {
  saxpy_(&n, &a, x, &incx, y, &incy);
}

void axpy(int n, const double *x, double *y, double a, int incx, int incy) {
  daxpy_(&n, &a, x, &incx, y, &incy);
}

void axpy(int n, const ::std::complex< float > *x, ::std::complex< float > *y,
          ::std::complex< float > a, int incx, int incy) {
  caxpy_(&n, &a, x, &incx, y, &incy);
}

void axpy(int n, const ::std::complex< double > *x, ::std::complex< double > *y,
          ::std::complex< double > a, int incx, int incy) {
  zaxpy_(&n, &a, x, &incx, y, &incy);
}

void copy(int n, const float *x, float *y, int incx, int incy) {
  scopy_(&n, x, &incx, y, &incy);
}

void copy(int n, const double *x, double *y, int incx, int incy) {
  dcopy_(&n, x, &incx, y, &incy);
}

void copy(int n, const ::std::complex< float > *x, ::std::complex< float > *y,
          int incx, int incy) {
  ccopy_(&n, x, &incx, y, &incy);
}

void copy(int n, const ::std::complex< double > *x, ::std::complex< double > *y,
          int incx, int incy) {
  zcopy_(&n, x, &incx, y, &incy);
}

float dot(int n, const float *x, const float *y, int incx, int incy) {
  return sdot_(&n, x, &incx, y, &incy);
}

double dot(int n, const double *x, const double *y, int incx, int incy) {
  return ddot_(&n, x, &incx, y, &incy);
}

::std::complex< float > dot(
     int n, const ::std::complex< float > *x, const ::std::complex< float > *y,
     int incx, int incy) {
  return dotu(n, x, y, incx, incy);
}

::std::complex< double > dot(
     int n, const ::std::complex< double > *x,
     const ::std::complex< double > *y, int incx, int incy) {
  return dotu(n, x, y, incx, incy);
}

float dotc(int n, const float *x, const float *y, int incx, int incy) {
  return dot(n, x, y, incx, incy);
}

double dotc(int n, const double *x, const double *y, int incx, int incy) {
  return dot(n, x, y, incx, incy);
}

::std::complex< float > dotc(
     int n, const ::std::complex< float > *x, const ::std::complex< float > *y,
     int incx, int incy) {
  ::std::complex< float > r;
  cdotcsubtd_(&r, &n, x, &incx, y, &incy);
  return r;
}

::std::complex< double > dotc(
     int n, const ::std::complex< double > *x,
     const ::std::complex< double > *y, int incx, int incy) {
  ::std::complex< double > r;
  zdotcsubtd_(&r, &n, x, &incx, y, &incy);
  return r;
}

float dotu(int n, const float *x, const float *y, int incx, int incy) {
  return dot(n, x, y, incx, incy);
}

double dotu(int n, const double *x, const double *y, int incx, int incy) {
  return dot(n, x, y, incx, incy);
}

::std::complex< float > dotu(
     int n, const ::std::complex< float > *x, const ::std::complex< float > *y,
     int incx, int incy) {
  ::std::complex< float > r;
  cdotusubtd_(&r, &n, x, &incx, y, &incy);
  return r;
}

::std::complex< double > dotu(
     int n, const ::std::complex< double > *x,
     const ::std::complex< double > *y, int incx, int incy) {
  ::std::complex< double > r;
  zdotusubtd_(&r, &n, x, &incx, y, &incy);
  return r;
}

float nrm2(int n, const float *x, int incx) { return snrm2_(&n, x, &incx); }

double nrm2(int n, const double *x, int incx) { return dnrm2_(&n, x, &incx); }

float nrm2(int n, const ::std::complex< float > *x, int incx) {
  return scnrm2_(&n, x, &incx);
}

double nrm2(int n, const ::std::complex< double > *x, int incx) {
  return dznrm2_(&n, x, &incx);
}

void rot(int n, float *x, float *y, const float c, const float s, int incx,
         int incy) {
  srot_(&n, x, &incx, y, &incy, &c, &s);
}

void rot(int n, double *x, double *y, const double c, const double s, int incx,
         int incy) {
  drot_(&n, x, &incx, y, &incy, &c, &s);
}

void rot(int n, ::std::complex< float > *x, ::std::complex< float > *y,
         const float c, const float s, int incx, int incy) {
  csrot_(&n, x, &incx, y, &incy, &c, &s);
}

void rot(int n, ::std::complex< double > *x, ::std::complex< double > *y,
         const double c, const double s, int incx, int incy) {
  zdrot_(&n, x, &incx, y, &incy, &c, &s);
}

void rotg(float *a, float *b, float *c, float *s) { srotg_(a, b, c, s); }

void rotg(double *a, double *b, double *c, double *s) { drotg_(a, b, c, s); }

void rotg(::std::complex< float > *a, ::std::complex< float > *b, float *c,
          ::std::complex< float > *s) {
  crotg_(a, b, c, s);
}

void rotg(::std::complex< double > *a, ::std::complex< double > *b, double *c,
          ::std::complex< double > *s) {
  zrotg_(a, b, c, s);
}

void rotm(int n, float *x, float *y, const float *param, int incx, int incy) {
  srotm_(&n, x, &incx, y, &incy, param);
}

void rotm(int n, double *x, double *y, const double *param, int incx,
          int incy) {
  drotm_(&n, x, &incx, y, &incy, param);
}

void rotmg(float *d1, float *d2, float *x1, const float x2,
           const float *param) {
  srotmg_(d1, d2, x1, &x2, param);
}

void rotmg(double *d1, double *d2, double *x1, const double x2,
           const double *param) {
  drotmg_(d1, d2, x1, &x2, param);
}

void scal(int n, float *x, float a, int incx) { sscal_(&n, &a, x, &incx); }

void scal(int n, double *x, double a, int incx) { dscal_(&n, &a, x, &incx); }

void scal(int n, ::std::complex< float > *x, ::std::complex< float > a,
          int incx) {
  cscal_(&n, &a, x, &incx);
}

void scal(int n, ::std::complex< double > *x, ::std::complex< double > a,
          int incx) {
  zscal_(&n, &a, x, &incx);
}

void scal(int n, ::std::complex< float > *x, float a, int incx) {
  csscal_(&n, &a, x, &incx);
}

void scal(int n, ::std::complex< double > *x, double a, int incx) {
  zdscal_(&n, &a, x, &incx);
}

void swap(int n, float *x, float *y, int incx, int incy) {
  sswap_(&n, x, &incx, y, &incy);
}

void swap(int n, double *x, double *y, int incx, int incy) {
  dswap_(&n, x, &incx, y, &incy);
}

void swap(int n, ::std::complex< float > *x, ::std::complex< float > *y,
          int incx, int incy) {
  cswap_(&n, x, &incx, y, &incy);
}

void swap(int n, ::std::complex< double > *x, ::std::complex< double > *y,
          int incx, int incy) {
  zswap_(&n, x, &incx, y, &incy);
}

int iamax(int n, const float *x, int incx) {
  int r = isamax_(&n, x, &incx);
  return (r ? r - 1 : 0);
}

int iamax(int n, const double *x, int incx) {
  int r = idamax_(&n, x, &incx);
  return (r ? r - 1 : 0);
}

int iamax(int n, const ::std::complex< float > *x, int incx) {
  int r = icamax_(&n, x, &incx);
  return (r ? r - 1 : 0);
}

int iamax(int n, const ::std::complex< double > *x, int incx) {
  int r = izamax_(&n, x, &incx);
  return (r ? r - 1 : 0);
}

}  // namespace cxxblas
}  // namespace linalg
}  // namespace thunder
