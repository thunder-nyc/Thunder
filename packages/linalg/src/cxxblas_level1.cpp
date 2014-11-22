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

float asum(const int n, const float *x, const int incx) {
  return sasum_(&n, x, &incx);
}

double asum(const int n, const double *x, const int incx) {
  return dasum_(&n, x, &incx);
}

float asum(const int n, const ::std::complex< float > *x, const int incx) {
  return scasum_(&n, x, &incx);
}

double asum(const int n, const ::std::complex< double > *x, const int incx) {
  return dzasum_(&n, x, &incx);
}

void axpy(const int n, const float *x, float *y, const float a, const int incx,
          const int incy) {
  saxpy_(&n, &a, x, &incx, y, &incy);
}

void axpy(const int n, const double *x, double *y, const double a,
          const int incx, const int incy) {
  daxpy_(&n, &a, x, &incx, y, &incy);
}

void axpy(const int n, const ::std::complex< float > *x,
          ::std::complex< float > *y, const ::std::complex< float > a,
          const int incx, const int incy) {
  caxpy_(&n, &a, x, &incx, y, &incy);
}

void axpy(const int n, const ::std::complex< double > *x,
          ::std::complex< double > *y, const ::std::complex< double > a,
          const int incx, const int incy) {
  zaxpy_(&n, &a, x, &incx, y, &incy);
}

void copy(const int n, const float *x, float *y, const int incx,
          const int incy) {
  scopy_(&n, x, &incx, y, &incy);
}

void copy(const int n, const double *x, double *y, const int incx,
          const int incy) {
  dcopy_(&n, x, &incx, y, &incy);
}

void copy(const int n, const ::std::complex< float > *x,
          ::std::complex< float > *y, const int incx, const int incy) {
  ccopy_(&n, x, &incx, y, &incy);
}

void copy(const int n, const ::std::complex< double > *x,
          ::std::complex< double > *y, const int incx, const int incy) {
  zcopy_(&n, x, &incx, y, &incy);
}

float dot(const int n, const float *x, const float *y, const int incx,
          const int incy) {
  return sdot_(&n, x, &incx, y, &incy);
}

double dot(const int n, const double *x, const double *y, const int incx,
           const int incy) {
  return ddot_(&n, x, &incx, y, &incy);
}

::std::complex< float > dot(
     const int n, const ::std::complex< float > *x,
     const ::std::complex< float > *y, const int incx, const int incy) {
  return dotu(n, x, y, incx, incy);
}

::std::complex< double > dot(
     const int n, const ::std::complex< double > *x,
     const ::std::complex< double > *y, const int incx, const int incy) {
  return dotu(n, x, y, incx, incy);
}

float dotc(const int n, const float *x, const float *y, const int incx,
           const int incy) {
  return dot(n, x, y, incx, incy);
}

double dotc(const int n, const double *x, const double *y, const int incx,
            const int incy) {
  return dot(n, x, y, incx, incy);
}

::std::complex< float > dotc(
     const int n, const ::std::complex< float > *x,
     const ::std::complex< float > *y, const int incx, const int incy) {
  ::std::complex< float > r;
  cdotcsubtd_(&r, &n, x, &incx, y, &incy);
  return r;
}

::std::complex< double > dotc(
     const int n, const ::std::complex< double > *x,
     const ::std::complex< double > *y, const int incx, const int incy) {
  ::std::complex< double > r;
  zdotcsubtd_(&r, &n, x, &incx, y, &incy);
  return r;
}

float dotu(const int n, const float *x, const float *y, const int incx,
           const int incy) {
  return dot(n, x, y, incx, incy);
}

double dotu(const int n, const double *x, const double *y, const int incx,
            const int incy) {
  return dot(n, x, y, incx, incy);
}

::std::complex< float > dotu(
     const int n, const ::std::complex< float > *x,
     const ::std::complex< float > *y, const int incx, const int incy) {
  ::std::complex< float > r;
  cdotusubtd_(&r, &n, x, &incx, y, &incy);
  return r;
}

::std::complex< double > dotu(
     const int n, const ::std::complex< double > *x,
     const ::std::complex< double > *y, const int incx, const int incy) {
  ::std::complex< double > r;
  zdotusubtd_(&r, &n, x, &incx, y, &incy);
  return r;
}

float nrm2(const int n, const float *x, const int incx) {
  return snrm2_(&n, x, &incx);
}

double nrm2(const int n, const double *x, const int incx) {
  return dnrm2_(&n, x, &incx);
}

float nrm2(const int n, const ::std::complex< float > *x, const int incx) {
  return scnrm2_(&n, x, &incx);
}

double nrm2(const int n, const ::std::complex< double > *x, const int incx) {
  return dznrm2_(&n, x, &incx);
}

void rot(const int n, float *x, float *y, const float c, const float s,
         const int incx, const int incy) {
  srot_(&n, x, &incx, y, &incy, &c, &s);
}

void rot(const int n, double *x, double *y, const double c, const double s,
         const int incx, const int incy) {
  drot_(&n, x, &incx, y, &incy, &c, &s);
}

void rot(const int n, ::std::complex< float > *x, ::std::complex< float > *y,
         const float c, const float s, const int incx, const int incy) {
  csrot_(&n, x, &incx, y, &incy, &c, &s);
}

void rot(const int n, ::std::complex< double > a, ::std::complex< double > *x,
         ::std::complex< double > *y, const double c, const double s,
         const int incx, const int incy) {
  zdrot_(&n, x, &incx, y, &incy, &c, &s);
}

void rotg(float *a, float *b, float *c, float *s) {
  srotg_(a, b, c, s);
}

void rotg(double *a, double *b, double *c, double *s) {
  drotg_(a, b, c, s);
}

void rotg(::std::complex< float > *a, ::std::complex< float > *b, float *c,
          ::std::complex< float > *s) {
  crotg_(a, b, c, s);
}

void rotg(::std::complex< double > *a, ::std::complex< double > *b, double *c,
          ::std::complex< double > *s) {
  zrotg_(a, b, c, s);
}

void rotm(const int n, float *x, float *y, const float *param, const int incx,
          const int incy) {
  srotm_(&n, x, &incx, y, &incy, param);
}

void rotm(const int n, double *x, double *y, const double *param,
          const int incx, const int incy) {
  drotm_(&n, x, &incx, y, &incy, param);
}

void scal(const int n, float *x, const float a, const int incx) {
  sscal_(&n, &a, x, &incx);
}

void scal(const int n, double *x, const double a, const int incx) {
  dscal_(&n, &a, x, &incx);
}

void scal(const int n, ::std::complex< float > *x,
          const ::std::complex< float > a, const int incx) {
  cscal_(&n, &a, x, &incx);
}

void scal(const int n, ::std::complex< double > *x,
          const ::std::complex< double > a, const int incx) {
  zscal_(&n, &a, x, &incx);
}

void scal(const int n, ::std::complex< float > *x, const float a,
          const int incx) {
  csscal_(&n, &a, x, &incx);
}

void scal(const int n, ::std::complex< double > *x, const double a,
          const int incx) {
  zdscal_(&n, &a, x, &incx);
}

void swap(const int n, float *x, float *y, const int incx, const int incy) {
  sswap_(&n, x, &incx, y, &incy);
}

void swap(const int n, double *x, double *y, const int incx, const int incy) {
  dswap_(&n, x, &incx, y, &incy);
}

void swap(const int n, ::std::complex< float > *x, ::std::complex< float > *y,
          const int incx, const int incy) {
  cswap_(&n, x, &incx, y, &incy);
}

void swap(const int n, ::std::complex< double > *x, ::std::complex< double > *y,
          const int incx, const int incy) {
  zswap_(&n, x, &incx, y, &incy);
}

int iamax(const int n, const float *x, const int incx) {
  return isamax_(&n, x, &incx);
}

int iamax(const int n, const double *x, const int incx) {
  return idamax_(&n, x, &incx);
}

int iamax(const int n, const ::std::complex< float > *x, const int incx) {
  return icamax_(&n, x, &incx);
}

int iamax(const int n, const ::std::complex< double > *x, const int incx) {
  return izamax_(&n, x, &incx);
}

}  // namespace cxxblas
}  // namespace linalg
}  // namespace thunder
