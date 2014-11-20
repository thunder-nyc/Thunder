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

#ifndef THUNDER_LINALG_CXXBLAS_HPP_
#define THUNDER_LINALG_CXXBLAS_HPP_

#include <complex>

#include "thunder/linalg/blas.hpp"

namespace thunder {
namespace linalg {
namespace cxxblas {

// BLAS types

enum class Order{
  kRowMajor = 101,
  kColMajor = 102
};
enum class Trans{
  kNoTrans = 111,
  kTrans = 112,
  kConjTrans = 113
};
enum class Uplo{
  kUpper = 121,
  kLower = 122
};
enum class Diag{
  kNonUnit = 131,
  kUnit = 132
};
enum class Side{
  kLeft = 141,
  kRight = 142
};


// BLAS level-1 functions

float asum(const int n, const float *x, const int incx = 1);
double asum(const int n, const double *x, const int incx = 1);
float asum(const int n, const ::std::complex< float > *x, const int incx = 1);
double asum(const int n, const ::std::complex< double > *x, const int incx = 1);

void axpy(const int n, float a, const float *x, float *y, const int incx = 1,
          const int incy = 1);
void axpy(const int n, double a, const double *x, double *y, const int incx = 1,
          const int incy = 1);
void axpy(const int n, ::std::complex< float > a,
          const ::std::complex< float > *x,
          ::std::complex< float > *y, const int incx = 1, const int incy = 1);
void axpy(const int n, ::std::complex< double > a,
          const ::std::complex< double > *x,
          ::std::complex< double > *y, const int incx = 1, const int incy = 1);

void copy(const int n, const float *x, const float *y, const int incx = 1,
          const int incy = 1);
void copy(const int n, const double *x, const double *y, const int incx = 1,
          const int incy = 1);
void copy(const int n, const ::std::complex< float > *x,
          ::std::complex< float > *y, const int incx = 1, const int incy = 1);
void copy(const int n, const ::std::complex< double > *x,
          ::std::complex< double > *y, const int incx = 1, const int incy = 1);

float dot(const int n, const float *x, const float *y, const int incx = 1,
          const int incy = 1);
double dot(const int n, const double *x, const double *y, const int incx = 1,
           const int incy = 1);
::std::complex< float > dot(
     const int n, const ::std::complex< float > *x,
     const ::std::complex< float > *y, const int incx = 1, const int incy = 1);
::std::complex< double > dot(
     const int n, const ::std::complex< double > *x,
     const ::std::complex< double > *y, const int incx = 1, const int incy = 1);

float dotc(const int n, const float *x, const float *y, const int incx = 1,
           const int incy = 1);
double dotc(const int n, const double *x, const double *y, const int incx = 1,
            const int incy = 1);
::std::complex< float > dotc(
     const int n, const ::std::complex< float > *x,
     const ::std::complex< float > *y, const int incx = 1, const int incy = 1);
::std::complex< double > dotc(
     const int n, const ::std::complex< double > *x,
     const ::std::complex< double > *y, const int incx = 1, const int incy = 1);

float nrm2(const int n, const float *x, const int incx = 1);
double nrm2(const int n, const double *x, const int incx = 1);
float nrm2(const int n, const ::std::complex< float > *x, const int incx = 1);
double nrm2(const int n, const ::std::complex< double > *x, const int incx = 1);

void rot(const int n, float a, float *x, float *y, float c, float s,
         const int incx = 1, const int incy = 1);
void rot(const int n, double a, double *x, double *y, double c, double s,
         const int incx = 1, const int incy = 1);
void rot(const int n, ::std::complex< float > a, ::std::complex< float > *x,
         ::std::complex< float > *y, const int incx = 1, const int incy = 1);
void rot(const int n, ::std::complex< double > a, ::std::complex< double > *x,
         ::std::complex< double > *y, const int incx = 1, const int incy = 1);

void rotg(float *a, float *b, float *c, float *s);
void rotg(double *a, double *b, double *c, double *s);
void rotg(::std::complex< float > *a, ::std::complex< float > *b, float *c,
          ::std::complex< float > *s);
void rotg(::std::complex< double > *a, ::std::complex< double > *b, double *c,
          ::std::complex< double > *s);

void rotm(const int n, float *x, float *y, const float *param,
          const int incx = 1, const int incy = 1);
void rotm(const int n, double *x, double *y, const double *param,
          const int incx = 1, const int incy = 1);

void rotmg(float *d1, float *d2, float *x1, const float *x2,
           const float *param);
void rotmg(double *d1, double *d2, float *x1, const double *x2,
           const double *param);

void scal(const int n, const float a, float *x, const int incx = 1);
void scal(const int n, const double a, double *x, const int incx = 1);
void scal(const int n, const ::std::complex< float > a,
          ::std::complex< float > *x, const int incx = 1);
void scal(const int n, const ::std::complex< double > a,
          ::std::complex< double > *x, const int incx = 1);
void scal(const int n, const float alpha, ::std::complex< float > *x,
          const int incx = 1);
void scal(const int n, const double alpha, ::std::complex< double > *x,
          const int incx = 1);

void swap(const int n, float *x, float *y, const int incx = 1,
            const int incy = 1);
void swap(const int n, double *x, double *y, const int incx = 1,
            const int incy = 1);
void swap(const int n, ::std::complex< float > *x, ::std::complex< float > *y,
            const int incx = 1, const int incy = 1);
void swap(const int n, ::std::complex< double > *x,
            ::std::complex< double > *y, const int incx = 1,
            const int incy = 1);

int iamax(const int n, const float *x, const int incx = 1);
int iamax(const int n, const double *x, const int incx = 1);
int iamax(const int n, const ::std::complex< float > *x, const int incx = 1);
int iamax(const int n, const ::std::complex< double > *x, const int incx = 1);

int iamin(const int n, const float *x, const int incx = 1);
int iamin(const int n, const double *x, const int incx = 1);
int iamin(const int n, const ::std::complex< float > *x, const int incx = 1);
int iamin(const int n, const ::std::complex< double > *x, const int incx = 1);

float cabs1(const ::std::complex< float > *x);
double cabs1(const ::std::complex< double > *x);


// BLAS level-2 functions

}  // namespace cxxblas
}  // namespace linalg
}  // namespace thunder

#endif  // THUNDER_LINALG_CXXBLAS_HPP_
