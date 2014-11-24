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

// CXXBLAS enumeration types

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


// CXXBLAS level-1 functions

float asum(const int n, const float *x, const int incx = 1);
double asum(const int n, const double *x, const int incx = 1);
float asum(const int n, const ::std::complex< float > *x, const int incx = 1);
double asum(const int n, const ::std::complex< double > *x, const int incx = 1);

void axpy(const int n, const float *x, float *y, const float a = 1.0,
          const int incx = 1, const int incy = 1);
void axpy(const int n, const double *x, double *y, const double a = 1.0,
          const int incx = 1, const int incy = 1);
void axpy(const int n, const ::std::complex< float > *x,
          ::std::complex< float > *y, const ::std::complex< float > a = 1.0,
          const int incx = 1, const int incy = 1);
void axpy(const int n, const ::std::complex< double > *x,
          ::std::complex< double > *y, const ::std::complex< double > a = 1.0,
          const int incx = 1, const int incy = 1);

void copy(const int n, const float *x, float *y, const int incx = 1,
          const int incy = 1);
void copy(const int n, const double *x, double *y, const int incx = 1,
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

float dotu(const int n, const float *x, const float *y, const int incx = 1,
           const int incy = 1);
double dotu(const int n, const double *x, const double *y, const int incx = 1,
            const int incy = 1);
::std::complex< float > dotu(
     const int n, const ::std::complex< float > *x,
     const ::std::complex< float > *y, const int incx = 1, const int incy = 1);
::std::complex< double > dotu(
     const int n, const ::std::complex< double > *x,
     const ::std::complex< double > *y, const int incx = 1, const int incy = 1);

float nrm2(const int n, const float *x, const int incx = 1);
double nrm2(const int n, const double *x, const int incx = 1);
float nrm2(const int n, const ::std::complex< float > *x, const int incx = 1);
double nrm2(const int n, const ::std::complex< double > *x, const int incx = 1);

void rot(const int n, float *x, float *y, const float c = 1.0,
         const float s = 1.0, const int incx = 1, const int incy = 1);
void rot(const int n, double *x, double *y, const double c = 1.0,
         const double s = 1.0, const int incx = 1, const int incy = 1);
void rot(const int n, ::std::complex< float > *x, ::std::complex< float > *y,
         const float c = 1.0, const float s = 1.0, const int incx = 1,
         const int incy = 1);
void rot(const int n, ::std::complex< double > a, ::std::complex< double > *x,
         ::std::complex< double > *y, const double c = 1.0,
         const double s = 1.0, const int incx = 1, const int incy = 1);

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

void scal(const int n, float *x, const float a, const int incx = 1);
void scal(const int n, double *x, const double a, const int incx = 1);
void scal(const int n, ::std::complex< float > *x,
          const ::std::complex< float > a, const int incx = 1);
void scal(const int n, ::std::complex< double > *x,
          const ::std::complex< double > a, const int incx = 1);
void scal(const int n, ::std::complex< float > *x, const float a,
          const int incx = 1);
void scal(const int n, ::std::complex< double > *x, const double a,
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


// CXXBLAS level-2 functions -- these functions are row major

void gbmv(const int m, const int n, const float *a, const float *x, float *y,
          const float alpha = 1.0, const float beta = 0.0, const int kl = 0,
          const int ku = 0, const int lda = 0, const int incx = 1,
          const int incy = 1, const Order order = Order::kRowMajor,
          const Trans trans = Trans::kNoTrans);
void gbmv(const int m, const int n,  const double *a, const double *x,
          double *y, const double alpha = 1.0, const double beta = 0.0,
          const int kl = 0, const int ku = 0, const int lda = 0,
          const int incx = 1, const int incy = 1,
          const Order order = Order::kRowMajor,
          const Trans trans = Trans::kNoTrans);
void gbmv(const int m, const int n,  const ::std::complex< float > *a,
          const ::std::complex< float > *x, ::std::complex< float > *y,
          const ::std::complex< float > alpha = 1.0,
          const ::std::complex< float > beta = 0.0, const int kl = 0,
          const int ku = 0, const int lda = 0, const int incx = 1,
          const int incy = 1, const Order order = Order::kRowMajor,
          const Trans trans = Trans::kNoTrans);
void gbmv(const int m, const int n, const ::std::complex< double > *a,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          const ::std::complex< double > alpha = 1.0,
          const ::std::complex< double > beta = 0.0, const int kl = 0,
          const int ku = 0, const int lda = 0, const int incx = 1,
          const int incy = 1, const Order order = Order::kRowMajor,
          const Trans trans = Trans::kNoTrans);

void gemv(const int m, const int n, const float *a, const float *x, float *y,
          const float alpha = 1.0, const float beta = 1.0, const int lda = 0,
          const int incx = 1, const int incy = 1,
          const Order order = Order::kRowMajor,
          const Trans trans = Trans::kNoTrans);
void gemv(const int m, const int n, const double *a, const double *x, double *y,
          const double alpha = 1.0, const double beta = 1.0, const int lda = 0,
          const int incx = 1, const int incy = 1,
          const Order order = Order::kRowMajor,
          const Trans trans = Trans::kNoTrans);
void gemv(const int m, const int n, const ::std::complex< float > *a,
          const ::std::complex< float > *x, ::std::complex< float > *y,
          const ::std::complex< float > alpha = 1.0,
          const ::std::complex< float > beta = 1.0, const int lda = 0,
          const int incx = 1, const int incy = 1,
          const Order order = Order::kRowMajor,
          const Trans trans = Trans::kNoTrans);
void gemv(const int m, const int n, const ::std::complex< double > *a,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          const ::std::complex< double > alpha = 1.0,
          const ::std::complex< double > beta = 1.0, const int lda = 0,
          const int incx = 1, const int incy = 1,
          const Order order = Order::kRowMajor,
          const Trans trans = Trans::kNoTrans);
void gemv(const int m, const int n, float *a, const ::std::complex< float > *x,
          ::std::complex< float > *y, const ::std::complex< float > alpha = 1.0,
          const ::std::complex< float > beta = 1.0, const int lda = 0,
          const int incx = 1, const int incy = 1,
          const Order order = Order::kRowMajor,
          const Trans trans = Trans::kNoTrans);
void gemv(const int m, const int n, double *a,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          const ::std::complex< double > alpha = 1.0,
          const ::std::complex< double > beta = 1.0, const int lda = 0,
          const int incx = 1, const int incy = 1,
          const Order order = Order::kRowMajor,
          const Trans trans = Trans::kNoTrans);

void ger(const int m, const int n, const float *x, const float *y, float *a,
         const float alpha = 1.0, const int incx = 1, const int incy = 1,
         const int lda = 0, const Order order = Order::kRowMajor);
void ger(const int m, const int n, const double *x, const double *y, double *a,
         const double alpha = 1.0, const int incx = 1, const int incy = 1,
         const int lda = 0, const Order order = Order::kRowMajor);
void ger(const int m, const int n, const ::std::complex< float > *x,
         const ::std::complex< float > *y, ::std::complex< float > *a,
         const ::std::complex< float > alpha = 1.0, const int incx = 1,
         const int incy = 1, const int lda = 0,
         const Order order = Order::kRowMajor);
void ger(const int m, const int n, const ::std::complex< double > *x,
         const ::std::complex< double > *y, ::std::complex< double > *a,
         const ::std::complex< double > alpha = 1.0, const int incx = 1,
         const int incy = 1, const int lda = 0,
         const Order order = Order::kRowMajor);

void gerc(const int m, const int n, const float *x, const float *y, float *a,
          const float alpha = 1.0, const int incx = 1, const int incy = 1,
          const int lda = 0, const Order order = Order::kRowMajor);
void gerc(const int m, const int n, const double *x, const double *y, double *a,
          const double alpha = 1.0, const int incx = 1, const int incy = 1,
          const int lda = 0, const Order order = Order::kRowMajor);
void gerc(const int m, const int n, const ::std::complex< float > *x,
          const ::std::complex< float > *y, ::std::complex< float > *a,
          const ::std::complex< float > alpha = 1.0, const int incx = 1,
          const int incy = 1, const int lda = 0,
          const Order order = Order::kRowMajor);
void gerc(const int m, const int n, const ::std::complex< double > *x,
          const ::std::complex< double > *y, ::std::complex< double > *a,
          const ::std::complex< double > alpha = 1.0, const int incx = 1,
          const int incy = 1, const int lda = 0,
          const Order order = Order::kRowMajor);

void geru(const int m, const int n, const float *x, const float *y, float *a,
          const float alpha = 1.0, const int incx = 1, const int incy = 1,
          const int lda = 0, const Order order = Order::kRowMajor);
void geru(const int m, const int n, const double *x, const double *y, double *a,
          const double alpha = 1.0, const int incx = 1, const int incy = 1,
          const int lda = 0, const Order order = Order::kRowMajor);
void geru(const int m, const int n, const ::std::complex< float > *x,
          const ::std::complex< float > *y, ::std::complex< float > *a,
          const ::std::complex< float > alpha = 1.0, const int incx = 1,
          const int incy = 1, const int lda = 0,
          const Order order = Order::kRowMajor);
void geru(const int m, const int n, const ::std::complex< double > *x,
          const ::std::complex< double > *y, ::std::complex< double > *a,
          const ::std::complex< double > alpha = 1.0, const int incx = 1,
          const int incy = 1, const int lda = 0,
          const Order order = Order::kRowMajor);

void hbmv(const int n, const float *a, const float *x, float *y,
          const float alpha = 1.0, const float beta = 0.0, const int k = 0,
          const int lda = 0, const int incx = 1, const int incy = 1,
          const Order order = Order::kRowMajor,
          const Uplo uplo = Uplo::kUpper);
void hbmv(const int n, const double *a, const double *x, double *y,
          const double alpha = 1.0, const float beta = 0.0, const int k = 0,
          const int lda = 0, const int incx = 1, const int incy = 1,
          const Order order = Order::kRowMajor,
          const Uplo uplo = Uplo::kUpper);
void hbmv(const int n, const ::std::complex< float > *a,
          const ::std::complex< float > *x, ::std::complex< float > *y,
          const ::std::complex< float > alpha = 1.0,
          const ::std::complex< float > beta = 0.0, const int k = 0,
          const int lda = 0, const int incx = 1, const int incy = 1,
          const Order order = Order::kRowMajor,
          const Uplo uplo = Uplo::kUpper);
void hbmv(const int n, const ::std::complex< double > *a,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          const ::std::complex< double > alpha = 1.0,
          const ::std::complex< double > beta = 0.0, const int k = 0,
          const int lda = 0, const int incx = 1, const int incy = 1,
          const Order order = Order::kRowMajor,
          const Uplo uplo = Uplo::kUpper);

void hemv(const int n, const float *a, const float *x, float *y,
          const float alpha = 1.0, const float beta = 0.0, const int lda = 0,
          const int incx = 1, const int incy = 1,
          const Order order = Order::kRowMajor,
          const Uplo uplo = Uplo::kUpper);
void hemv(const int n, const double *a, const double *x, double *y,
          const double alpha = 1.0, const float beta = 0.0, const int lda = 0,
          const int incx = 1, const int incy = 1,
          const Order order = Order::kRowMajor,
          const Uplo uplo = Uplo::kUpper);
void hemv(const int n, const ::std::complex< float > *a,
          const ::std::complex< float > *x, ::std::complex< float > *y,
          const ::std::complex< float > alpha = 1.0,
          const ::std::complex< float > beta = 0.0, const int lda = 0,
          const int incx = 1, const int incy = 1,
          const Order order = Order::kRowMajor,
          const Uplo uplo = Uplo::kUpper);
void hemv(const int n, const ::std::complex< double > *a,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          const ::std::complex< double > alpha = 1.0,
          const ::std::complex< double > beta = 0.0, const int lda = 0,
          const int incx = 1, const int incy = 1,
          const Order order = Order::kRowMajor,
          const Uplo uplo = Uplo::kUpper);

void her(const int n, const float *x, float *a, const float alpha = 1.0,
         const int incx = 1, const int lda = 0,
         const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper);
void her(const int n, const double *x, double *a, const double alpha = 1.0,
         const int incx = 1, const int lda = 0,
         const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper);
void her(const int n, const ::std::complex< float > *x,
         ::std::complex< float > *a, const ::std::complex< float > alpha = 1.0,
         const int incx = 1, const int lda = 0,
         const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper);
void her(const int n, const ::std::complex< double > *x,
         ::std::complex< double > *a,
         const ::std::complex< double > alpha = 1.0, const int incx = 1,
         const int lda = 0, const Order order = Order::kRowMajor,
         const Uplo uplo = Uplo::kUpper);

void her2(const int n, const float *x, const float *y, float *a,
          const float alpha = 1.0, const int incx = 1, const int incy = 1,
          const int lda = 0, const Order order = Order::kRowMajor,
          const Uplo uplo = Uplo::kUpper);
void her2(const int n, const double *x, const double *y, double *a,
          const double alpha = 1.0, const int incx = 1, const int incy = 1,
          const int lda = 0, const Order order = Order::kRowMajor,
          const Uplo uplo = Uplo::kUpper);
void her2(const int n, const ::std::complex< float > *x,
          const ::std::complex< float > *y, ::std::complex< float > *a,
          const ::std::complex< float > alpha = 1.0, const int incx = 1,
          const int incy = 1, const int lda = 0,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper);
void her2(const int n, const ::std::complex< double > *x,
          const ::std::complex< double > *y, ::std::complex< double > *a,
          const ::std::complex< double > alpha = 1.0, const int incx = 1,
          const int incy = 1, const int lda = 0,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper);

void hpmv(const int n, const float *ap, const float *x, float *y,
          const float alpha = 1.0, const float beta = 0.0, const int incx = 1,
          const int incy = 1, const Order order = Order::kRowMajor,
          const Uplo uplo = Uplo::kUpper);
void hpmv(const int n, const double *ap, const double *x, double *y,
          const double alpha = 1.0, const double beta = 0.0, const int incx = 1,
          const int incy = 1, const Order order = Order::kRowMajor,
          const Uplo uplo = Uplo::kUpper);
void hpmv(const int n, const ::std::complex< float > *ap,
          const ::std::complex< float > *x, ::std::complex< float > *y,
          const ::std::complex< float > alpha = 1.0,
          const ::std::complex< float > beta = 0.0, const int incx = 1,
          const int incy = 1, const Order order = Order::kRowMajor,
          const Uplo uplo = Uplo::kUpper);
void hpmv(const int n, const ::std::complex< double > *ap,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          const ::std::complex< double > alpha = 1.0,
          const ::std::complex< double > beta = 0.0, const int incx = 1,
          const int incy = 1, const Order order = Order::kRowMajor,
          const Uplo uplo = Uplo::kUpper);

void hpr(const int n, const float *x, float *a, const float alpha = 1.0,
         const int incx = 1, const Order order = Order::kRowMajor,
         const Uplo uplo = Uplo::kUpper);
void hpr(const int n, const double *x, double *a, const double alpha = 1.0,
         const int incx = 1, const Order order = Order::kRowMajor,
         const Uplo uplo = Uplo::kUpper);
void hpr(const int n, const ::std::complex< float > *x,
         ::std::complex< float > *a, const ::std::complex< float > alpha = 1.0,
         const int incx = 1, const Order order = Order::kRowMajor,
         const Uplo uplo = Uplo::kUpper);
void hpr(const int n, const ::std::complex< double > *x,
         ::std::complex< double > *a,
         const ::std::complex< double > alpha = 1.0,
         const int incx = 1, const Order order = Order::kRowMajor,
         const Uplo uplo = Uplo::kUpper);

void hpr2(const int n, const float *x, const float *y, float *a,
          const float alpha = 1.0, const int incx = 1, const int incy = 1,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper);
void hpr2(const int n, const double *x, const double *y, double *a,
          const double alpha = 1.0, const int incx = 1, const int incy = 1,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper);
void hpr2(const int n, const ::std::complex< float > *x,
          const ::std::complex< float > *y, ::std::complex< float > *a,
          const ::std::complex< float > alpha = 1.0, const int incx = 1,
          const int incy = 1, const Order order = Order::kRowMajor,
          const Uplo uplo = Uplo::kUpper);
void hpr2(const int n, const ::std::complex< double > *x,
          const ::std::complex< double > *y, ::std::complex< double > *a,
          const ::std::complex< double > alpha = 1.0, const int incx = 1,
          const int incy = 1, const Order order = Order::kRowMajor,
          const Uplo uplo = Uplo::kUpper);

void sbmv(const int n, const float *a, const float *x, float *y,
          const float alpha = 1.0, const float beta = 0.0, const int k = 0,
          const int lda = 0, const int incx = 1, const int incy = 1,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper);
void sbmv(const int n, const double *a, const double *x, double *y,
          const double alpha = 1.0, const float beta = 0.0, const int k = 0,
          const int lda = 0, const int incx = 1, const int incy = 1,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper);
void sbmv(const int n, const ::std::complex< float > *a,
          const ::std::complex< float > *x, ::std::complex< float > *y,
          const ::std::complex< float > alpha = 1.0,
          const ::std::complex< float > beta = 0.0, const int k = 0,
          const int lda = 0, const int incx = 1, const int incy = 1,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper);
void sbmv(const int n, const ::std::complex< double > *a,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          const ::std::complex< double > alpha = 1.0,
          const ::std::complex< double > beta = 0.0, const int k = 0,
          const int lda = 0, const int incx = 1, const int incy = 1,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper);

void spmv(const int n, const float *ap, const float *x, float *y,
          const float alpha = 1.0, const float beta = 0.0, const int incx = 1,
          const int incy = 1, const Order order = Order::kRowMajor,
          const Uplo uplo = Uplo::kUpper);
void spmv(const int n, const double *ap, const double *x, double *y,
          const double alpha = 1.0, const double beta = 0.0, const int incx = 1,
          const int incy = 1, const Order order = Order::kRowMajor,
          const Uplo uplo = Uplo::kUpper);
void spmv(const int n, const ::std::complex< float > *ap,
          const ::std::complex< float > *x, ::std::complex< float > *y,
          const ::std::complex< float > alpha = 1.0,
          const ::std::complex< float > beta = 0.0, const int incx = 1,
          const int incy = 1, const Order order = Order::kRowMajor,
          const Uplo uplo = Uplo::kUpper);
void spmv(const int n, const ::std::complex< double > *ap,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          const ::std::complex< double > alpha = 1.0,
          const ::std::complex< double > beta = 0.0, const int incx = 1,
          const int incy = 1, const Order order = Order::kRowMajor,
          const Uplo uplo = Uplo::kUpper);

void spr(const int n, const float *x, float *a, const float alpha = 1.0,
         const int incx = 1, const Order order = Order::kRowMajor,
         const Uplo uplo = Uplo::kUpper);
void spr(const int n, const double *x, double *a, const double alpha = 1.0,
         const int incx = 1, const Order order = Order::kRowMajor,
         const Uplo uplo = Uplo::kUpper);
void spr(const int n, const ::std::complex< float > *x,
         ::std::complex< float > *a, const ::std::complex< float > alpha = 1.0,
         const int incx = 1, const Order order = Order::kRowMajor,
         const Uplo uplo = Uplo::kUpper);
void spr(const int n, const ::std::complex< double > *x,
         ::std::complex< double > *a,
         const ::std::complex< double > alpha = 1.0, const int incx = 1,
         const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper);

void spr2(const int n, const float *x, const float *y, float *a,
          const float alpha = 1.0, const int incx = 1, const int incy = 1,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper);
void spr2(const int n, const double *x, const double *y, double *a,
          const double alpha = 1.0, const int incx = 1, const int incy = 1,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper);
void spr2(const int n, const ::std::complex< float > *x,
          const ::std::complex< float > *y, ::std::complex< float > *a,
          const ::std::complex< float > alpha = 1.0, const int incx = 1,
          const int incy = 1, const Order order = Order::kRowMajor,
          const Uplo uplo = Uplo::kUpper);
void spr2(const int n, const ::std::complex< double > *x,
          const ::std::complex< double > *y, ::std::complex< double > *a,
          const ::std::complex< double > alpha = 1.0, const int incx = 1,
          const int incy = 1, const Order order = Order::kRowMajor,
          const Uplo uplo = Uplo::kUpper);

void symv(const int n, const float *a, const float *x, float *y,
          const float alpha = 1.0, const float beta = 0.0, const int lda = 0,
          const int incx = 1, const int incy = 1,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper);
void symv(const int n, const double *a, const double *x, double *y,
          const double alpha = 1.0, const float beta = 0.0, const int lda = 0,
          const int incx = 1, const int incy = 1,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper);
void symv(const int n, const ::std::complex< float > *a,
          const ::std::complex< float > *x, ::std::complex< float > *y,
          const ::std::complex< float > alpha = 1.0,
          const ::std::complex< float > beta = 0.0, const int lda = 0,
          const int incx = 1, const int incy = 1,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper);
void symv(const int n, const ::std::complex< double > *a,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          const ::std::complex< double > alpha = 1.0,
          const ::std::complex< double > beta = 0.0, const int lda = 0,
          const int incx = 1, const int incy = 1,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper);

void syr(const int n, const float *x, float *a, const float alpha = 1.0,
         const int incx = 1, const int lda = 0,
         const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper);
void syr(const int n, const double *x, double *a, const double alpha = 1.0,
         const int incx = 1, const int lda = 0,
         const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper);
void syr(const int n, const ::std::complex< float > *x,
         ::std::complex< float > *a, const ::std::complex< float > alpha = 1.0,
         const int incx = 1, const int lda = 0,
         const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper);
void syr(const int n, const ::std::complex< double > *x,
         ::std::complex< double > *a,
         const ::std::complex< double > alpha = 1.0, const int incx = 1,
         const int lda = 0, const Order order = Order::kRowMajor,
         const Uplo uplo = Uplo::kUpper);

void syr2(const int n, const float *x, const float *y, float *a,
          const float alpha = 1.0, const int incx = 1, const int incy = 1,
          const int lda = 0, const Order order = Order::kRowMajor,
          const Uplo uplo = Uplo::kUpper);
void syr2(const int n, const double *x, const double *y, double *a,
          const double alpha = 1.0, const int incx = 1, const int incy = 1,
          const int lda = 0, const Order order = Order::kRowMajor,
          const Uplo uplo = Uplo::kUpper);
void syr2(const int n, const ::std::complex< float > *x,
          const ::std::complex< float > *y, ::std::complex< float > *a,
          const ::std::complex< float > alpha = 1.0, const int incx = 1,
          const int incy = 1, const int lda = 0,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper);
void syr2(const int n, const ::std::complex< double > *x,
          const ::std::complex< double > *y, ::std::complex< double > *a,
          const ::std::complex< double > alpha = 1.0, const int incx = 1,
          const int incy = 1, const int lda = 0,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper);

void tbmv(const int n, const float *a, float *x, const int k = 0,
          const int lda = 0, const int incx = 1, const Uplo uplo = Uplo::kUpper,
          const Order order = Order::kRowMajor,
          const Trans trans = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);
void tbmv(const int n, const double *a, double *x, const int k = 0,
          const int lda = 0, const int incx = 1, const Uplo uplo = Uplo::kUpper,
          const Order order = Order::kRowMajor,
          const Trans trans = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);
void tbmv(const int n, const ::std::complex< float > *a,
          ::std::complex< float > *x, const int k = 0, const int lda = 0,
          const int incx = 1, const Uplo uplo = Uplo::kUpper,
          const Order order = Order::kRowMajor,
          const Trans trans = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);
void tbmv(const int n, const ::std::complex< double > *a,
          ::std::complex< double > *x, const int k = 0, const int lda = 0,
          const int incx = 1, const Uplo uplo = Uplo::kUpper,
          const Order order = Order::kRowMajor,
          const Trans trans = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);

void tbsv(const int n, const float *a, float *x, const int k = 0,
          const int lda = 0, const int incx = 1, const Uplo uplo = Uplo::kUpper,
          const Order order = Order::kRowMajor,
          const Trans trans = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);
void tbsv(const int n, const double *a, double *x, const int k = 0,
          const int lda = 0, const int incx = 1, const Uplo uplo = Uplo::kUpper,
          const Order order = Order::kRowMajor,
          const Trans trans = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);
void tbsv(const int n, const ::std::complex< float > *a,
          ::std::complex< float > *x, const int k = 0, const int lda = 0,
          const int incx = 1, const Uplo uplo = Uplo::kUpper,
          const Order order = Order::kRowMajor,
          const Trans trans = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);
void tbsv(const int n, const ::std::complex< double > *a,
          ::std::complex< double > *x, const int k = 0, const int lda = 0,
          const int incx = 1, const Uplo uplo = Uplo::kUpper,
          const Order order = Order::kRowMajor,
          const Trans trans = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);

void tpmv(const int n, const float *ap, float *x, const int incx = 1,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper,
          const Trans trans = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);
void tpmv(const int n, const double *ap, double *x, const int incx = 1,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper,
          const Trans trans = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);
void tpmv(const int n, const ::std::complex< float > *ap,
          ::std::complex< float > *x, const int incx = 1,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper,
          const Trans trans = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);
void tpmv(const int n, const ::std::complex< double > *ap,
          ::std::complex< double > *x, const int incx = 1,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper,
          const Trans trans = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);

void tpsv(const int n, const float *ap, float *x, const int incx = 1,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper,
          const Trans trans = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);
void tpsv(const int n, const double *ap, double *x, const int incx = 1,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper,
          const Trans trans = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);
void tpsv(const int n, const ::std::complex< float > *ap,
          ::std::complex< float > *x, const int incx = 1,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper,
          const Trans trans = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);
void tpsv(const int n, const ::std::complex< double > *ap,
          ::std::complex< double > *x, const int incx = 1,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper,
          const Trans trans = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);

void trmv(const int n, const float *a, float *x, const int lda = 0,
          const int incx = 1, const Order order = Order::kRowMajor,
          const Uplo uplo = Uplo::kUpper, const Trans trans = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);
void trmv(const int n, const double *a, double *x, const int lda = 0,
          const int incx = 1, const Order order = Order::kRowMajor,
          const Uplo uplo = Uplo::kUpper, const Trans trans = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);
void trmv(const int n, const ::std::complex< float > *a,
          ::std::complex< float > *x, const int lda = 0, const int incx = 1,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper,
          const Trans trans = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);
void trmv(const int n, const ::std::complex< double > *a,
          ::std::complex< double > *x, const int lda = 0, const int incx = 1,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper,
          const Trans trans = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);

void trsv(const int n, const float *a, float *x, const int lda = 0,
          const int incx = 1, const Order order = Order::kRowMajor,
          const Uplo uplo = Uplo::kUpper, const Trans trans = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);
void trsv(const int n, const double *a, double *x, const int lda = 0,
          const int incx = 1, const Order order = Order::kRowMajor,
          const Uplo uplo = Uplo::kUpper, const Trans trans = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);
void trsv(const int n, const ::std::complex< float > *a,
          ::std::complex< float > *x, const int lda = 0, const int incx = 1,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper,
          const Trans trans = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);
void trsv(const int n, const ::std::complex< double > *a,
          ::std::complex< double > *x, const int lda = 0, const int incx = 1,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper,
          const Trans trans = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);


// CXXBLAS level-3 functions]

void gemm(const int m, const int n, const int k, const float *a, const float *b,
          float *c, const float alpha = 1.0, const float beta = 0.0,
          const int lda = 1, const int ldb = 1, const int ldc = 1,
          const Order order = Order::kRowMajor,
          const Trans transa = Trans::kNoTrans,
          const Trans transb = Trans::kNoTrans);
void gemm(const int m, const int n, const int k, const double *a,
          const double *b, double *c, const double alpha = 1.0,
          const double beta = 0.0, const int lda = 1, const int ldb = 1,
          const int ldc = 1, const Order order = Order::kRowMajor,
          const Trans transa = Trans::kNoTrans,
          const Trans transb = Trans::kNoTrans);
void gemm(const int m, const int n, const int k,
          const ::std::complex< float > *a, const ::std::complex< float > *b,
          ::std::complex< float > *c, const ::std::complex< float > alpha = 1.0,
          const ::std::complex< float > beta = 0.0, const int lda = 1,
          const int ldb = 1, const int ldc = 1,
          const Order order = Order::kRowMajor,
          const Trans transa = Trans::kNoTrans,
          const Trans transb = Trans::kNoTrans);
void gemm(const int m, const int n, const int k,
          const ::std::complex< double > *a, const ::std::complex< double > *b,
          ::std::complex< double > *c,
          const ::std::complex< double > alpha = 1.0,
          const ::std::complex< double > beta = 0.0, const int lda = 1,
          const int ldb = 1, const int ldc = 1,
          const Order order = Order::kRowMajor,          
          const Trans transa = Trans::kNoTrans,
          const Trans transb = Trans::kNoTrans);
void gemm(const int m, const int n, const int k,
          const float *a, const ::std::complex< float > *b,
          ::std::complex< float > *c, const ::std::complex< float > alpha = 1.0,
          const ::std::complex< float > beta = 0.0, const int lda = 1,
          const int ldb = 1, const int ldc = 1,
          const Order order = Order::kRowMajor,
          const Trans transa = Trans::kNoTrans,
          const Trans transb = Trans::kNoTrans);
void gemm(const int m, const int n, const int k,
          const double *a, const ::std::complex< double > *b,
          ::std::complex< double > *c,
          const ::std::complex< double > alpha = 1.0,
          const ::std::complex< double > beta = 0.0, const int lda = 1,
          const int ldb = 1, const int ldc = 1,
          const Order order = Order::kRowMajor,
          const Trans transa = Trans::kNoTrans,
          const Trans transb = Trans::kNoTrans);

void hemm(const int m, const int n, const float *a, const float *b, float *c,
          const float alpha = 1.0, const float beta = 0.0, const int lda = 1,
          const int ldb = 1, const int ldc = 1,
          const Order order = Order::kRowMajor, const Side side = Side::kLeft,
          const Uplo Uplo = Uplo::kUpper);
void hemm(const int m, const int n, const double *a, const double *b, double *c,
          const double alpha = 1.0, const double beta = 0.0, const int lda = 1,
          const int ldb = 1, const int ldc = 1,
          const Order order = Order::kRowMajor, const Side side = Side::kLeft,
          const Uplo Uplo = Uplo::kUpper);
void hemm(const int m, const int n, const ::std::complex< float > *a,
          const ::std::complex< float > *b, ::std::complex< float > *c,
          const ::std::complex< float > alpha = 1.0,
          const ::std::complex< float > beta = 0.0, const int lda = 1,
          const int ldb = 1, const int ldc = 1,
          const Order order = Order::kRowMajor, const Side side = Side::kLeft,
          const Uplo Uplo = Uplo::kUpper);
void hemm(const int m, const int n, const ::std::complex< double > *a,
          const ::std::complex< double > *b, ::std::complex< double > *c,
          const ::std::complex< double > alpha = 1.0,
          const ::std::complex< double > beta = 0.0, const int lda = 1,
          const int ldb = 1, const int ldc = 1,
          const Order order = Order::kRowMajor, const Side side = Side::kLeft,
          const Uplo Uplo = Uplo::kUpper);

void herk(const int n, const int k, const float *a, const float *c,
          const float alpha = 1.0, const float beta = 0.0,
          const int lda = 0, const int ldc = 0,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper,
          const Trans trans = Trans::kNoTrans);
void herk(const int n, const int k, const double *a, const double *c,
          const double alpha = 1.0, const double beta = 0.0,
          const int lda = 0, const int ldc = 0,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper,
          const Trans trans = Trans::kNoTrans);
void herk(const int n, const int k, const ::std::complex< float > *a,
          const ::std::complex< float > *c,
          const ::std::complex< float > alpha = 1.0,
          const ::std::complex< float > beta = 0.0,
          const int lda = 0, const int ldc = 0,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper,
          const Trans trans = Trans::kNoTrans);
void herk(const int n, const int k, const ::std::complex< double > *a,
          const ::std::complex< double > *c,
          const ::std::complex< double > alpha = 1.0,
          const ::std::complex< double > beta = 0.0,
          const int lda = 0, const int ldc = 0,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper,
          const Trans trans = Trans::kNoTrans);

void her2k(const int n, const int k, const float *a, const float *b,
           const float *c,  const float alpha = 1.0, const float beta = 0.0,
           const int lda = 0, const int ldb = 0, const int ldc = 0,
           const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper,
           const Trans trans = Trans::kNoTrans);
void her2k(const int n, const int k, const double *a, const float *b,
           const double *c,  const double alpha = 1.0, const double beta = 0.0,
           const int lda = 0, const int ldb = 0, const int ldc = 0,
           const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper,
           const Trans trans = Trans::kNoTrans);
void her2k(const int n, const int k, const ::std::complex< float > *a,
           const ::std::complex< float > *b,
           const ::std::complex< float > *c,
           const ::std::complex< float > alpha = 1.0,
           const ::std::complex< float > beta = 0.0,
           const int lda = 0, const int ldb = 0, const int ldc = 0,
           const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper,
           const Trans trans = Trans::kNoTrans);
void her2k(const int n, const int k, const ::std::complex< double > *a,
           const ::std::complex< double > *b,
           const ::std::complex< double > *c,
           const ::std::complex< double > alpha = 1.0,
           const ::std::complex< double > beta = 0.0,
           const int lda = 0, const int ldb = 0, const int ldc = 0,
           const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper,
           const Trans trans = Trans::kNoTrans);

void symm(const int m, const int n, const float *a, const float *b, float *c,
          const float alpha = 1.0, const float beta = 0.0, const int lda = 1,
          const int ldb = 1, const int ldc = 1,
          const Order order = Order::kRowMajor, const Side side = Side::kLeft,
          const Uplo Uplo = Uplo::kUpper);
void symm(const int m, const int n, const double *a, const double *b, double *c,
          const double alpha = 1.0, const double beta = 0.0, const int lda = 1,
          const int ldb = 1, const int ldc = 1,
          const Order order = Order::kRowMajor, const Side side = Side::kLeft,
          const Uplo Uplo = Uplo::kUpper);
void symm(const int m, const int n, const ::std::complex< float > *a,
          const ::std::complex< float > *b, ::std::complex< float > *c,
          const ::std::complex< float > alpha = 1.0,
          const ::std::complex< float > beta = 0.0, const int lda = 1,
          const int ldb = 1, const int ldc = 1,
          const Order order = Order::kRowMajor, const Side side = Side::kLeft,
          const Uplo Uplo = Uplo::kUpper);
void symm(const int m, const int n, const ::std::complex< double > *a,
          const ::std::complex< double > *b, ::std::complex< double > *c,
          const ::std::complex< double > alpha = 1.0,
          const ::std::complex< double > beta = 0.0, const int lda = 1,
          const int ldb = 1, const int ldc = 1,
          const Order order = Order::kRowMajor, const Side side = Side::kLeft,
          const Uplo Uplo = Uplo::kUpper);

void syrk(const int n, const int k, const float *a, const float *c,
          const float alpha = 1.0, const float beta = 0.0,
          const int lda = 0, const int ldc = 0,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper,
          const Trans trans = Trans::kNoTrans);
void syrk(const int n, const int k, const double *a, const double *c,
          const double alpha = 1.0, const double beta = 0.0,
          const int lda = 0, const int ldc = 0,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper,
          const Trans trans = Trans::kNoTrans);
void syrk(const int n, const int k, const ::std::complex< float > *a,
          const ::std::complex< float > *c,
          const ::std::complex< float > alpha = 1.0,
          const ::std::complex< float > beta = 0.0,
          const int lda = 0, const int ldc = 0,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper,
          const Trans trans = Trans::kNoTrans);
void syrk(const int n, const int k, const ::std::complex< double > *a,
          const ::std::complex< double > *c,
          const ::std::complex< double > alpha = 1.0,
          const ::std::complex< double > beta = 0.0,
          const int lda = 0, const int ldc = 0,
          const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper,
          const Trans trans = Trans::kNoTrans);

void syr2k(const int n, const int k, const float *a, const float *b,
           const float *c,  const float alpha = 1.0, const float beta = 0.0,
           const int lda = 0, const int ldb = 0, const int ldc = 0,
           const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper,
           const Trans trans = Trans::kNoTrans);
void syr2k(const int n, const int k, const double *a, const float *b,
           const double *c,  const double alpha = 1.0, const double beta = 0.0,
           const int lda = 0, const int ldb = 0, const int ldc = 0,
           const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper,
           const Trans trans = Trans::kNoTrans);
void syr2k(const int n, const int k, const ::std::complex< float > *a,
           const ::std::complex< float > *b,
           const ::std::complex< float > *c,
           const ::std::complex< float > alpha = 1.0,
           const ::std::complex< float > beta = 0.0,
           const int lda = 0, const int ldb = 0, const int ldc = 0,
           const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper,
           const Trans trans = Trans::kNoTrans);
void syr2k(const int n, const int k, const ::std::complex< double > *a,
           const ::std::complex< double > *b,
           const ::std::complex< double > *c,
           const ::std::complex< double > alpha = 1.0,
           const ::std::complex< double > beta = 0.0,
           const int lda = 0, const int ldb = 0, const int ldc = 0,
           const Order order = Order::kRowMajor, const Uplo uplo = Uplo::kUpper,
           const Trans trans = Trans::kNoTrans);

void trmm(const int m, const int n, const float *a, float *b,
          const float alpha = 1.0, const int lda = 0, const int ldb = 0,
          const Order order = Order::kRowMajor, const Side side = Side::kLeft,
          const Uplo uplo = Uplo::kUpper, const Trans transa = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);
void trmm(const int m, const int n, const double *a, double *b,
          const double alpha = 1.0, const int lda = 0, const int ldb = 0,
          const Order order = Order::kRowMajor, const Side side = Side::kLeft,
          const Uplo uplo = Uplo::kUpper, const Trans transa = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);
void trmm(const int m, const int n, const ::std::complex< float > *a,
          ::std::complex< float > *b,
          const ::std::complex< float > alpha = 1.0, const int lda = 0,
          const int ldb = 0, const Order order = Order::kRowMajor,
          const Side side = Side::kLeft, const Uplo uplo = Uplo::kUpper,
          const Trans transa = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);
void trmm(const int m, const int n, const ::std::complex< double > *a,
          ::std::complex< double > *b,
          const ::std::complex< double > alpha = 1.0, const int lda = 0,
          const int ldb = 0, const Order order = Order::kRowMajor,
          const Side side = Side::kLeft, const Uplo uplo = Uplo::kUpper,
          const Trans transa = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);

void trsm(const int m, const int n, const float *a, float *b,
          const float alpha = 1.0, const int lda = 0, const int ldb = 0,
          const Order order = Order::kRowMajor, const Side side = Side::kLeft,
          const Uplo uplo = Uplo::kUpper, const Trans transa = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);
void trsm(const int m, const int n, const double *a, double *b,
          const double alpha = 1.0, const int lda = 0, const int ldb = 0,
          const Order order = Order::kRowMajor, const Side side = Side::kLeft,
          const Uplo uplo = Uplo::kUpper, const Trans transa = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);
void trsm(const int m, const int n, const ::std::complex< float > *a,
          ::std::complex< float > *b,
          const ::std::complex< float > alpha = 1.0, const int lda = 0,
          const int ldb = 0, const Order order = Order::kRowMajor,
          const Side side = Side::kLeft, const Uplo uplo = Uplo::kUpper,
          const Trans transa = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);
void trsm(const int m, const int n, const ::std::complex< double > *a,
          ::std::complex< double > *b,
          const ::std::complex< double > alpha = 1.0, const int lda = 0,
          const int ldb = 0, const Order order = Order::kRowMajor,
          const Side side = Side::kLeft, const Uplo uplo = Uplo::kUpper,
          const Trans transa = Trans::kNoTrans,
          const Diag diag = Diag::kNonUnit);

}  // namespace cxxblas
}  // namespace linalg
}  // namespace thunder

#endif  // THUNDER_LINALG_CXXBLAS_HPP_
