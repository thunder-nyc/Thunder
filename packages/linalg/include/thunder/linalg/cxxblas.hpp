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

enum class Order { kRowMajor = 101, kColMajor = 102 };
enum class Trans { kNoTrans = 111, kTrans = 112, kConjTrans = 113 };
enum class Uplo { kUpper = 121, kLower = 122 };
enum class Diag { kNonUnit = 131, kUnit = 132 };
enum class Side { kLeft = 141, kRight = 142 };

// CXXBLAS level-1 functions

float asum(int n, const float *x, int incx = 1);
double asum(int n, const double *x, int incx = 1);
float asum(int n, const ::std::complex< float > *x, int incx = 1);
double asum(int n, const ::std::complex< double > *x, int incx = 1);

void axpy(int n, const float *x, float *y, float a = 1.0, int incx = 1,
          int incy = 1);
void axpy(int n, const double *x, double *y, double a = 1.0, int incx = 1,
          int incy = 1);
void axpy(int n, const ::std::complex< float > *x, ::std::complex< float > *y,
          ::std::complex< float > a = 1.0, int incx = 1, int incy = 1);
void axpy(int n, const ::std::complex< double > *x, ::std::complex< double > *y,
          ::std::complex< double > a = 1.0, int incx = 1, int incy = 1);

void copy(int n, const float *x, float *y, int incx = 1, int incy = 1);
void copy(int n, const double *x, double *y, int incx = 1, int incy = 1);
void copy(int n, const ::std::complex< float > *x, ::std::complex< float > *y,
          int incx = 1, int incy = 1);
void copy(int n, const ::std::complex< double > *x, ::std::complex< double > *y,
          int incx = 1, int incy = 1);

float dot(int n, const float *x, const float *y, int incx = 1, int incy = 1);
double dot(int n, const double *x, const double *y, int incx = 1, int incy = 1);
::std::complex< float > dot(
     int n, const ::std::complex< float > *x, const ::std::complex< float > *y,
     int incx = 1, int incy = 1);
::std::complex< double > dot(
     int n, const ::std::complex< double > *x,
     const ::std::complex< double > *y, int incx = 1, int incy = 1);

float dotc(int n, const float *x, const float *y, int incx = 1, int incy = 1);
double dotc(int n, const double *x, const double *y, int incx = 1,
            int incy = 1);
::std::complex< float > dotc(
     int n, const ::std::complex< float > *x, const ::std::complex< float > *y,
     int incx = 1, int incy = 1);
::std::complex< double > dotc(
     int n, const ::std::complex< double > *x,
     const ::std::complex< double > *y, int incx = 1, int incy = 1);

float dotu(int n, const float *x, const float *y, int incx = 1, int incy = 1);
double dotu(int n, const double *x, const double *y, int incx = 1,
            int incy = 1);
::std::complex< float > dotu(
     int n, const ::std::complex< float > *x, const ::std::complex< float > *y,
     int incx = 1, int incy = 1);
::std::complex< double > dotu(
     int n, const ::std::complex< double > *x,
     const ::std::complex< double > *y, int incx = 1, int incy = 1);

float nrm2(int n, const float *x, int incx = 1);
double nrm2(int n, const double *x, int incx = 1);
float nrm2(int n, const ::std::complex< float > *x, int incx = 1);
double nrm2(int n, const ::std::complex< double > *x, int incx = 1);

void rot(int n, float *x, float *y, const float c = 1.0, const float s = 1.0,
         int incx = 1, int incy = 1);
void rot(int n, double *x, double *y, const double c = 1.0,
         const double s = 1.0, int incx = 1, int incy = 1);
void rot(int n, ::std::complex< float > *x, ::std::complex< float > *y,
         const float c = 1.0, const float s = 1.0, int incx = 1, int incy = 1);
void rot(int n, ::std::complex< double > a, ::std::complex< double > *x,
         ::std::complex< double > *y, const double c = 1.0,
         const double s = 1.0, int incx = 1, int incy = 1);

void rotg(float *a, float *b, float *c, float *s);
void rotg(double *a, double *b, double *c, double *s);
void rotg(::std::complex< float > *a, ::std::complex< float > *b, float *c,
          ::std::complex< float > *s);
void rotg(::std::complex< double > *a, ::std::complex< double > *b, double *c,
          ::std::complex< double > *s);

void rotm(int n, float *x, float *y, const float *param, int incx = 1,
          int incy = 1);
void rotm(int n, double *x, double *y, const double *param, int incx = 1,
          int incy = 1);

void rotmg(float *d1, float *d2, float *x1, const float *x2,
           const float *param);
void rotmg(double *d1, double *d2, float *x1, const double *x2,
           const double *param);

void scal(int n, float *x, float a, int incx = 1);
void scal(int n, double *x, double a, int incx = 1);
void scal(int n, ::std::complex< float > *x, ::std::complex< float > a,
          int incx = 1);
void scal(int n, ::std::complex< double > *x, ::std::complex< double > a,
          int incx = 1);
void scal(int n, ::std::complex< float > *x, float a, int incx = 1);
void scal(int n, ::std::complex< double > *x, double a, int incx = 1);

void swap(int n, float *x, float *y, int incx = 1, int incy = 1);
void swap(int n, double *x, double *y, int incx = 1, int incy = 1);
void swap(int n, ::std::complex< float > *x, ::std::complex< float > *y,
          int incx = 1, int incy = 1);
void swap(int n, ::std::complex< double > *x, ::std::complex< double > *y,
          int incx = 1, int incy = 1);

int iamax(int n, const float *x, int incx = 1);
int iamax(int n, const double *x, int incx = 1);
int iamax(int n, const ::std::complex< float > *x, int incx = 1);
int iamax(int n, const ::std::complex< double > *x, int incx = 1);

// CXXBLAS level-2 functions -- these functions are row major

void gbmv(int m, int n, const float *a, const float *x, float *y,
          float alpha = 1.0, float beta = 0.0, int kl = 0, int ku = 0,
          int lda = 0, int incx = 1, int incy = 1,
          Order order = Order::kRowMajor, Trans trans = Trans::kNoTrans);
void gbmv(int m, int n, const double *a, const double *x, double *y,
          double alpha = 1.0, double beta = 0.0, int kl = 0, int ku = 0,
          int lda = 0, int incx = 1, int incy = 1,
          Order order = Order::kRowMajor, Trans trans = Trans::kNoTrans);
void gbmv(int m, int n, const ::std::complex< float > *a,
          const ::std::complex< float > *x, ::std::complex< float > *y,
          ::std::complex< float > alpha = 1.0,
          ::std::complex< float > beta = 0.0, int kl = 0, int ku = 0,
          int lda = 0, int incx = 1, int incy = 1,
          Order order = Order::kRowMajor, Trans trans = Trans::kNoTrans);
void gbmv(int m, int n, const ::std::complex< double > *a,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          ::std::complex< double > alpha = 1.0,
          ::std::complex< double > beta = 0.0, int kl = 0, int ku = 0,
          int lda = 0, int incx = 1, int incy = 1,
          Order order = Order::kRowMajor, Trans trans = Trans::kNoTrans);

void gemv(int m, int n, const float *a, const float *x, float *y,
          float alpha = 1.0, float beta = 1.0, int lda = 0, int incx = 1,
          int incy = 1, Order order = Order::kRowMajor,
          Trans trans = Trans::kNoTrans);
void gemv(int m, int n, const double *a, const double *x, double *y,
          double alpha = 1.0, double beta = 1.0, int lda = 0, int incx = 1,
          int incy = 1, Order order = Order::kRowMajor,
          Trans trans = Trans::kNoTrans);
void gemv(int m, int n, const ::std::complex< float > *a,
          const ::std::complex< float > *x, ::std::complex< float > *y,
          ::std::complex< float > alpha = 1.0,
          ::std::complex< float > beta = 1.0, int lda = 0, int incx = 1,
          int incy = 1, Order order = Order::kRowMajor,
          Trans trans = Trans::kNoTrans);
void gemv(int m, int n, const ::std::complex< double > *a,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          ::std::complex< double > alpha = 1.0,
          ::std::complex< double > beta = 1.0, int lda = 0, int incx = 1,
          int incy = 1, Order order = Order::kRowMajor,
          Trans trans = Trans::kNoTrans);

void ger(int m, int n, const float *x, const float *y, float *a,
         float alpha = 1.0, int incx = 1, int incy = 1, int lda = 0,
         Order order = Order::kRowMajor);
void ger(int m, int n, const double *x, const double *y, double *a,
         double alpha = 1.0, int incx = 1, int incy = 1, int lda = 0,
         Order order = Order::kRowMajor);
void ger(int m, int n, const ::std::complex< float > *x,
         const ::std::complex< float > *y, ::std::complex< float > *a,
         ::std::complex< float > alpha = 1.0, int incx = 1, int incy = 1,
         int lda = 0, Order order = Order::kRowMajor);
void ger(int m, int n, const ::std::complex< double > *x,
         const ::std::complex< double > *y, ::std::complex< double > *a,
         ::std::complex< double > alpha = 1.0, int incx = 1, int incy = 1,
         int lda = 0, Order order = Order::kRowMajor);

void gerc(int m, int n, const float *x, const float *y, float *a,
          float alpha = 1.0, int incx = 1, int incy = 1, int lda = 0,
          Order order = Order::kRowMajor);
void gerc(int m, int n, const double *x, const double *y, double *a,
          double alpha = 1.0, int incx = 1, int incy = 1, int lda = 0,
          Order order = Order::kRowMajor);
void gerc(int m, int n, const ::std::complex< float > *x,
          const ::std::complex< float > *y, ::std::complex< float > *a,
          ::std::complex< float > alpha = 1.0, int incx = 1, int incy = 1,
          int lda = 0, Order order = Order::kRowMajor);
void gerc(int m, int n, const ::std::complex< double > *x,
          const ::std::complex< double > *y, ::std::complex< double > *a,
          ::std::complex< double > alpha = 1.0, int incx = 1, int incy = 1,
          int lda = 0, Order order = Order::kRowMajor);

void geru(int m, int n, const float *x, const float *y, float *a,
          float alpha = 1.0, int incx = 1, int incy = 1, int lda = 0,
          Order order = Order::kRowMajor);
void geru(int m, int n, const double *x, const double *y, double *a,
          double alpha = 1.0, int incx = 1, int incy = 1, int lda = 0,
          Order order = Order::kRowMajor);
void geru(int m, int n, const ::std::complex< float > *x,
          const ::std::complex< float > *y, ::std::complex< float > *a,
          ::std::complex< float > alpha = 1.0, int incx = 1, int incy = 1,
          int lda = 0, Order order = Order::kRowMajor);
void geru(int m, int n, const ::std::complex< double > *x,
          const ::std::complex< double > *y, ::std::complex< double > *a,
          ::std::complex< double > alpha = 1.0, int incx = 1, int incy = 1,
          int lda = 0, Order order = Order::kRowMajor);

void hbmv(int n, const float *a, const float *x, float *y, float alpha = 1.0,
          float beta = 0.0, int k = 0, int lda = 0, int incx = 1, int incy = 1,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void hbmv(int n, const double *a, const double *x, double *y,
          double alpha = 1.0, float beta = 0.0, int k = 0, int lda = 0,
          int incx = 1, int incy = 1, Order order = Order::kRowMajor,
          Uplo uplo = Uplo::kUpper);
void hbmv(int n, const ::std::complex< float > *a,
          const ::std::complex< float > *x, ::std::complex< float > *y,
          ::std::complex< float > alpha = 1.0,
          ::std::complex< float > beta = 0.0, int k = 0, int lda = 0,
          int incx = 1, int incy = 1, Order order = Order::kRowMajor,
          Uplo uplo = Uplo::kUpper);
void hbmv(int n, const ::std::complex< double > *a,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          ::std::complex< double > alpha = 1.0,
          ::std::complex< double > beta = 0.0, int k = 0, int lda = 0,
          int incx = 1, int incy = 1, Order order = Order::kRowMajor,
          Uplo uplo = Uplo::kUpper);

void hemv(int n, const float *a, const float *x, float *y, float alpha = 1.0,
          float beta = 0.0, int lda = 0, int incx = 1, int incy = 1,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void hemv(int n, const double *a, const double *x, double *y,
          double alpha = 1.0, float beta = 0.0, int lda = 0, int incx = 1,
          int incy = 1, Order order = Order::kRowMajor,
          Uplo uplo = Uplo::kUpper);
void hemv(int n, const ::std::complex< float > *a,
          const ::std::complex< float > *x, ::std::complex< float > *y,
          ::std::complex< float > alpha = 1.0,
          ::std::complex< float > beta = 0.0, int lda = 0, int incx = 1,
          int incy = 1, Order order = Order::kRowMajor,
          Uplo uplo = Uplo::kUpper);
void hemv(int n, const ::std::complex< double > *a,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          ::std::complex< double > alpha = 1.0,
          ::std::complex< double > beta = 0.0, int lda = 0, int incx = 1,
          int incy = 1, Order order = Order::kRowMajor,
          Uplo uplo = Uplo::kUpper);

void her(int n, const float *x, float *a, float alpha = 1.0, int incx = 1,
         int lda = 0, Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void her(int n, const double *x, double *a, double alpha = 1.0, int incx = 1,
         int lda = 0, Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void her(int n, const ::std::complex< float > *x, ::std::complex< float > *a,
         ::std::complex< float > alpha = 1.0, int incx = 1, int lda = 0,
         Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void her(int n, const ::std::complex< double > *x, ::std::complex< double > *a,
         ::std::complex< double > alpha = 1.0, int incx = 1, int lda = 0,
         Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);

void her2(int n, const float *x, const float *y, float *a, float alpha = 1.0,
          int incx = 1, int incy = 1, int lda = 0,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void her2(int n, const double *x, const double *y, double *a,
          double alpha = 1.0, int incx = 1, int incy = 1, int lda = 0,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void her2(int n, const ::std::complex< float > *x,
          const ::std::complex< float > *y, ::std::complex< float > *a,
          ::std::complex< float > alpha = 1.0, int incx = 1, int incy = 1,
          int lda = 0, Order order = Order::kRowMajor,
          Uplo uplo = Uplo::kUpper);
void her2(int n, const ::std::complex< double > *x,
          const ::std::complex< double > *y, ::std::complex< double > *a,
          ::std::complex< double > alpha = 1.0, int incx = 1, int incy = 1,
          int lda = 0, Order order = Order::kRowMajor,
          Uplo uplo = Uplo::kUpper);

void hpmv(int n, const float *ap, const float *x, float *y, float alpha = 1.0,
          float beta = 0.0, int incx = 1, int incy = 1,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void hpmv(int n, const double *ap, const double *x, double *y,
          double alpha = 1.0, double beta = 0.0, int incx = 1, int incy = 1,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void hpmv(int n, const ::std::complex< float > *ap,
          const ::std::complex< float > *x, ::std::complex< float > *y,
          ::std::complex< float > alpha = 1.0,
          ::std::complex< float > beta = 0.0, int incx = 1, int incy = 1,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void hpmv(int n, const ::std::complex< double > *ap,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          ::std::complex< double > alpha = 1.0,
          ::std::complex< double > beta = 0.0, int incx = 1, int incy = 1,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);

void hpr(int n, const float *x, float *a, float alpha = 1.0, int incx = 1,
         Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void hpr(int n, const double *x, double *a, double alpha = 1.0, int incx = 1,
         Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void hpr(int n, const ::std::complex< float > *x, ::std::complex< float > *a,
         ::std::complex< float > alpha = 1.0, int incx = 1,
         Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void hpr(int n, const ::std::complex< double > *x, ::std::complex< double > *a,
         ::std::complex< double > alpha = 1.0, int incx = 1,
         Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);

void hpr2(int n, const float *x, const float *y, float *a, float alpha = 1.0,
          int incx = 1, int incy = 1, Order order = Order::kRowMajor,
          Uplo uplo = Uplo::kUpper);
void hpr2(int n, const double *x, const double *y, double *a,
          double alpha = 1.0, int incx = 1, int incy = 1,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void hpr2(int n, const ::std::complex< float > *x,
          const ::std::complex< float > *y, ::std::complex< float > *a,
          ::std::complex< float > alpha = 1.0, int incx = 1, int incy = 1,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void hpr2(int n, const ::std::complex< double > *x,
          const ::std::complex< double > *y, ::std::complex< double > *a,
          ::std::complex< double > alpha = 1.0, int incx = 1, int incy = 1,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);

void sbmv(int n, const float *a, const float *x, float *y, float alpha = 1.0,
          float beta = 0.0, int k = 0, int lda = 0, int incx = 1, int incy = 1,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void sbmv(int n, const double *a, const double *x, double *y,
          double alpha = 1.0, float beta = 0.0, int k = 0, int lda = 0,
          int incx = 1, int incy = 1, Order order = Order::kRowMajor,
          Uplo uplo = Uplo::kUpper);
void sbmv(int n, const ::std::complex< float > *a,
          const ::std::complex< float > *x, ::std::complex< float > *y,
          ::std::complex< float > alpha = 1.0,
          ::std::complex< float > beta = 0.0, int k = 0, int lda = 0,
          int incx = 1, int incy = 1, Order order = Order::kRowMajor,
          Uplo uplo = Uplo::kUpper);
void sbmv(int n, const ::std::complex< double > *a,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          ::std::complex< double > alpha = 1.0,
          ::std::complex< double > beta = 0.0, int k = 0, int lda = 0,
          int incx = 1, int incy = 1, Order order = Order::kRowMajor,
          Uplo uplo = Uplo::kUpper);

void spmv(int n, const float *ap, const float *x, float *y, float alpha = 1.0,
          float beta = 0.0, int incx = 1, int incy = 1,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void spmv(int n, const double *ap, const double *x, double *y,
          double alpha = 1.0, double beta = 0.0, int incx = 1, int incy = 1,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void spmv(int n, const ::std::complex< float > *ap,
          const ::std::complex< float > *x, ::std::complex< float > *y,
          ::std::complex< float > alpha = 1.0,
          ::std::complex< float > beta = 0.0, int incx = 1, int incy = 1,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void spmv(int n, const ::std::complex< double > *ap,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          ::std::complex< double > alpha = 1.0,
          ::std::complex< double > beta = 0.0, int incx = 1, int incy = 1,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);

void spr(int n, const float *x, float *a, float alpha = 1.0, int incx = 1,
         Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void spr(int n, const double *x, double *a, double alpha = 1.0, int incx = 1,
         Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void spr(int n, const ::std::complex< float > *x, ::std::complex< float > *a,
         ::std::complex< float > alpha = 1.0, int incx = 1,
         Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void spr(int n, const ::std::complex< double > *x, ::std::complex< double > *a,
         ::std::complex< double > alpha = 1.0, int incx = 1,
         Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);

void spr2(int n, const float *x, const float *y, float *a, float alpha = 1.0,
          int incx = 1, int incy = 1, Order order = Order::kRowMajor,
          Uplo uplo = Uplo::kUpper);
void spr2(int n, const double *x, const double *y, double *a,
          double alpha = 1.0, int incx = 1, int incy = 1,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void spr2(int n, const ::std::complex< float > *x,
          const ::std::complex< float > *y, ::std::complex< float > *a,
          ::std::complex< float > alpha = 1.0, int incx = 1, int incy = 1,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void spr2(int n, const ::std::complex< double > *x,
          const ::std::complex< double > *y, ::std::complex< double > *a,
          ::std::complex< double > alpha = 1.0, int incx = 1, int incy = 1,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);

void symv(int n, const float *a, const float *x, float *y, float alpha = 1.0,
          float beta = 0.0, int lda = 0, int incx = 1, int incy = 1,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void symv(int n, const double *a, const double *x, double *y,
          double alpha = 1.0, float beta = 0.0, int lda = 0, int incx = 1,
          int incy = 1, Order order = Order::kRowMajor,
          Uplo uplo = Uplo::kUpper);
void symv(int n, const ::std::complex< float > *a,
          const ::std::complex< float > *x, ::std::complex< float > *y,
          ::std::complex< float > alpha = 1.0,
          ::std::complex< float > beta = 0.0, int lda = 0, int incx = 1,
          int incy = 1, Order order = Order::kRowMajor,
          Uplo uplo = Uplo::kUpper);
void symv(int n, const ::std::complex< double > *a,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          ::std::complex< double > alpha = 1.0,
          ::std::complex< double > beta = 0.0, int lda = 0, int incx = 1,
          int incy = 1, Order order = Order::kRowMajor,
          Uplo uplo = Uplo::kUpper);

void syr(int n, const float *x, float *a, float alpha = 1.0, int incx = 1,
         int lda = 0, Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void syr(int n, const double *x, double *a, double alpha = 1.0, int incx = 1,
         int lda = 0, Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void syr(int n, const ::std::complex< float > *x, ::std::complex< float > *a,
         ::std::complex< float > alpha = 1.0, int incx = 1, int lda = 0,
         Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void syr(int n, const ::std::complex< double > *x, ::std::complex< double > *a,
         ::std::complex< double > alpha = 1.0, int incx = 1, int lda = 0,
         Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);

void syr2(int n, const float *x, const float *y, float *a, float alpha = 1.0,
          int incx = 1, int incy = 1, int lda = 0,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void syr2(int n, const double *x, const double *y, double *a,
          double alpha = 1.0, int incx = 1, int incy = 1, int lda = 0,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper);
void syr2(int n, const ::std::complex< float > *x,
          const ::std::complex< float > *y, ::std::complex< float > *a,
          ::std::complex< float > alpha = 1.0, int incx = 1, int incy = 1,
          int lda = 0, Order order = Order::kRowMajor,
          Uplo uplo = Uplo::kUpper);
void syr2(int n, const ::std::complex< double > *x,
          const ::std::complex< double > *y, ::std::complex< double > *a,
          ::std::complex< double > alpha = 1.0, int incx = 1, int incy = 1,
          int lda = 0, Order order = Order::kRowMajor,
          Uplo uplo = Uplo::kUpper);

void tbmv(int n, const float *a, float *x, int k = 0, int lda = 0, int incx = 1,
          Uplo uplo = Uplo::kUpper, Order order = Order::kRowMajor,
          Trans trans = Trans::kNoTrans, Diag diag = Diag::kNonUnit);
void tbmv(int n, const double *a, double *x, int k = 0, int lda = 0,
          int incx = 1, Uplo uplo = Uplo::kUpper,
          Order order = Order::kRowMajor, Trans trans = Trans::kNoTrans,
          Diag diag = Diag::kNonUnit);
void tbmv(int n, const ::std::complex< float > *a, ::std::complex< float > *x,
          int k = 0, int lda = 0, int incx = 1, Uplo uplo = Uplo::kUpper,
          Order order = Order::kRowMajor, Trans trans = Trans::kNoTrans,
          Diag diag = Diag::kNonUnit);
void tbmv(int n, const ::std::complex< double > *a, ::std::complex< double > *x,
          int k = 0, int lda = 0, int incx = 1, Uplo uplo = Uplo::kUpper,
          Order order = Order::kRowMajor, Trans trans = Trans::kNoTrans,
          Diag diag = Diag::kNonUnit);

void tbsv(int n, const float *a, float *x, int k = 0, int lda = 0, int incx = 1,
          Uplo uplo = Uplo::kUpper, Order order = Order::kRowMajor,
          Trans trans = Trans::kNoTrans, Diag diag = Diag::kNonUnit);
void tbsv(int n, const double *a, double *x, int k = 0, int lda = 0,
          int incx = 1, Uplo uplo = Uplo::kUpper,
          Order order = Order::kRowMajor, Trans trans = Trans::kNoTrans,
          Diag diag = Diag::kNonUnit);
void tbsv(int n, const ::std::complex< float > *a, ::std::complex< float > *x,
          int k = 0, int lda = 0, int incx = 1, Uplo uplo = Uplo::kUpper,
          Order order = Order::kRowMajor, Trans trans = Trans::kNoTrans,
          Diag diag = Diag::kNonUnit);
void tbsv(int n, const ::std::complex< double > *a, ::std::complex< double > *x,
          int k = 0, int lda = 0, int incx = 1, Uplo uplo = Uplo::kUpper,
          Order order = Order::kRowMajor, Trans trans = Trans::kNoTrans,
          Diag diag = Diag::kNonUnit);

void tpmv(int n, const float *ap, float *x, int incx = 1,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper,
          Trans trans = Trans::kNoTrans, Diag diag = Diag::kNonUnit);
void tpmv(int n, const double *ap, double *x, int incx = 1,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper,
          Trans trans = Trans::kNoTrans, Diag diag = Diag::kNonUnit);
void tpmv(int n, const ::std::complex< float > *ap, ::std::complex< float > *x,
          int incx = 1, Order order = Order::kRowMajor,
          Uplo uplo = Uplo::kUpper, Trans trans = Trans::kNoTrans,
          Diag diag = Diag::kNonUnit);
void tpmv(int n, const ::std::complex< double > *ap,
          ::std::complex< double > *x, int incx = 1,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper,
          Trans trans = Trans::kNoTrans, Diag diag = Diag::kNonUnit);

void tpsv(int n, const float *ap, float *x, int incx = 1,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper,
          Trans trans = Trans::kNoTrans, Diag diag = Diag::kNonUnit);
void tpsv(int n, const double *ap, double *x, int incx = 1,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper,
          Trans trans = Trans::kNoTrans, Diag diag = Diag::kNonUnit);
void tpsv(int n, const ::std::complex< float > *ap, ::std::complex< float > *x,
          int incx = 1, Order order = Order::kRowMajor,
          Uplo uplo = Uplo::kUpper, Trans trans = Trans::kNoTrans,
          Diag diag = Diag::kNonUnit);
void tpsv(int n, const ::std::complex< double > *ap,
          ::std::complex< double > *x, int incx = 1,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper,
          Trans trans = Trans::kNoTrans, Diag diag = Diag::kNonUnit);

void trmv(int n, const float *a, float *x, int lda = 0, int incx = 1,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper,
          Trans trans = Trans::kNoTrans, Diag diag = Diag::kNonUnit);
void trmv(int n, const double *a, double *x, int lda = 0, int incx = 1,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper,
          Trans trans = Trans::kNoTrans, Diag diag = Diag::kNonUnit);
void trmv(int n, const ::std::complex< float > *a, ::std::complex< float > *x,
          int lda = 0, int incx = 1, Order order = Order::kRowMajor,
          Uplo uplo = Uplo::kUpper, Trans trans = Trans::kNoTrans,
          Diag diag = Diag::kNonUnit);
void trmv(int n, const ::std::complex< double > *a, ::std::complex< double > *x,
          int lda = 0, int incx = 1, Order order = Order::kRowMajor,
          Uplo uplo = Uplo::kUpper, Trans trans = Trans::kNoTrans,
          Diag diag = Diag::kNonUnit);

void trsv(int n, const float *a, float *x, int lda = 0, int incx = 1,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper,
          Trans trans = Trans::kNoTrans, Diag diag = Diag::kNonUnit);
void trsv(int n, const double *a, double *x, int lda = 0, int incx = 1,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper,
          Trans trans = Trans::kNoTrans, Diag diag = Diag::kNonUnit);
void trsv(int n, const ::std::complex< float > *a, ::std::complex< float > *x,
          int lda = 0, int incx = 1, Order order = Order::kRowMajor,
          Uplo uplo = Uplo::kUpper, Trans trans = Trans::kNoTrans,
          Diag diag = Diag::kNonUnit);
void trsv(int n, const ::std::complex< double > *a, ::std::complex< double > *x,
          int lda = 0, int incx = 1, Order order = Order::kRowMajor,
          Uplo uplo = Uplo::kUpper, Trans trans = Trans::kNoTrans,
          Diag diag = Diag::kNonUnit);

// CXXBLAS level-3 functions]

void gemm(int m, int n, int k, const float *a, const float *b, float *c,
          float alpha = 1.0, float beta = 0.0, int lda = 1, int ldb = 1,
          int ldc = 1, Order order = Order::kRowMajor,
          Trans transa = Trans::kNoTrans, Trans transb = Trans::kNoTrans);
void gemm(int m, int n, int k, const double *a, const double *b, double *c,
          double alpha = 1.0, double beta = 0.0, int lda = 1, int ldb = 1,
          int ldc = 1, Order order = Order::kRowMajor,
          Trans transa = Trans::kNoTrans, Trans transb = Trans::kNoTrans);
void gemm(int m, int n, int k, const ::std::complex< float > *a,
          const ::std::complex< float > *b, ::std::complex< float > *c,
          ::std::complex< float > alpha = 1.0,
          ::std::complex< float > beta = 0.0, int lda = 1, int ldb = 1,
          int ldc = 1, Order order = Order::kRowMajor,
          Trans transa = Trans::kNoTrans, Trans transb = Trans::kNoTrans);
void gemm(int m, int n, int k, const ::std::complex< double > *a,
          const ::std::complex< double > *b, ::std::complex< double > *c,
          ::std::complex< double > alpha = 1.0,
          ::std::complex< double > beta = 0.0, int lda = 1, int ldb = 1,
          int ldc = 1, Order order = Order::kRowMajor,
          Trans transa = Trans::kNoTrans, Trans transb = Trans::kNoTrans);
void gemm(int m, int n, int k, const float *a, const ::std::complex< float > *b,
          ::std::complex< float > *c, ::std::complex< float > alpha = 1.0,
          ::std::complex< float > beta = 0.0, int lda = 1, int ldb = 1,
          int ldc = 1, Order order = Order::kRowMajor,
          Trans transa = Trans::kNoTrans, Trans transb = Trans::kNoTrans);
void gemm(int m, int n, int k, const double *a,
          const ::std::complex< double > *b, ::std::complex< double > *c,
          ::std::complex< double > alpha = 1.0,
          ::std::complex< double > beta = 0.0, int lda = 1, int ldb = 1,
          int ldc = 1, Order order = Order::kRowMajor,
          Trans transa = Trans::kNoTrans, Trans transb = Trans::kNoTrans);

void hemm(int m, int n, const float *a, const float *b, float *c,
          float alpha = 1.0, float beta = 0.0, int lda = 1, int ldb = 1,
          int ldc = 1, Order order = Order::kRowMajor, Side side = Side::kLeft,
          Uplo Uplo = Uplo::kUpper);
void hemm(int m, int n, const double *a, const double *b, double *c,
          double alpha = 1.0, double beta = 0.0, int lda = 1, int ldb = 1,
          int ldc = 1, Order order = Order::kRowMajor, Side side = Side::kLeft,
          Uplo Uplo = Uplo::kUpper);
void hemm(int m, int n, const ::std::complex< float > *a,
          const ::std::complex< float > *b, ::std::complex< float > *c,
          ::std::complex< float > alpha = 1.0,
          ::std::complex< float > beta = 0.0, int lda = 1, int ldb = 1,
          int ldc = 1, Order order = Order::kRowMajor, Side side = Side::kLeft,
          Uplo Uplo = Uplo::kUpper);
void hemm(int m, int n, const ::std::complex< double > *a,
          const ::std::complex< double > *b, ::std::complex< double > *c,
          ::std::complex< double > alpha = 1.0,
          ::std::complex< double > beta = 0.0, int lda = 1, int ldb = 1,
          int ldc = 1, Order order = Order::kRowMajor, Side side = Side::kLeft,
          Uplo Uplo = Uplo::kUpper);

void herk(int n, int k, const float *a, const float *c, float alpha = 1.0,
          float beta = 0.0, int lda = 0, int ldc = 0,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper,
          Trans trans = Trans::kNoTrans);
void herk(int n, int k, const double *a, const double *c, double alpha = 1.0,
          double beta = 0.0, int lda = 0, int ldc = 0,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper,
          Trans trans = Trans::kNoTrans);
void herk(int n, int k, const ::std::complex< float > *a,
          const ::std::complex< float > *c, ::std::complex< float > alpha = 1.0,
          ::std::complex< float > beta = 0.0, int lda = 0, int ldc = 0,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper,
          Trans trans = Trans::kNoTrans);
void herk(int n, int k, const ::std::complex< double > *a,
          const ::std::complex< double > *c,
          ::std::complex< double > alpha = 1.0,
          ::std::complex< double > beta = 0.0, int lda = 0, int ldc = 0,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper,
          Trans trans = Trans::kNoTrans);

void her2k(int n, int k, const float *a, const float *b, const float *c,
           float alpha = 1.0, float beta = 0.0, int lda = 0, int ldb = 0,
           int ldc = 0, Order order = Order::kRowMajor,
           Uplo uplo = Uplo::kUpper, Trans trans = Trans::kNoTrans);
void her2k(int n, int k, const double *a, const float *b, const double *c,
           double alpha = 1.0, double beta = 0.0, int lda = 0, int ldb = 0,
           int ldc = 0, Order order = Order::kRowMajor,
           Uplo uplo = Uplo::kUpper, Trans trans = Trans::kNoTrans);
void her2k(int n, int k, const ::std::complex< float > *a,
           const ::std::complex< float > *b, const ::std::complex< float > *c,
           ::std::complex< float > alpha = 1.0,
           ::std::complex< float > beta = 0.0, int lda = 0, int ldb = 0,
           int ldc = 0, Order order = Order::kRowMajor,
           Uplo uplo = Uplo::kUpper, Trans trans = Trans::kNoTrans);
void her2k(int n, int k, const ::std::complex< double > *a,
           const ::std::complex< double > *b, const ::std::complex< double > *c,
           ::std::complex< double > alpha = 1.0,
           ::std::complex< double > beta = 0.0, int lda = 0, int ldb = 0,
           int ldc = 0, Order order = Order::kRowMajor,
           Uplo uplo = Uplo::kUpper, Trans trans = Trans::kNoTrans);

void symm(int m, int n, const float *a, const float *b, float *c,
          float alpha = 1.0, float beta = 0.0, int lda = 1, int ldb = 1,
          int ldc = 1, Order order = Order::kRowMajor, Side side = Side::kLeft,
          Uplo Uplo = Uplo::kUpper);
void symm(int m, int n, const double *a, const double *b, double *c,
          double alpha = 1.0, double beta = 0.0, int lda = 1, int ldb = 1,
          int ldc = 1, Order order = Order::kRowMajor, Side side = Side::kLeft,
          Uplo Uplo = Uplo::kUpper);
void symm(int m, int n, const ::std::complex< float > *a,
          const ::std::complex< float > *b, ::std::complex< float > *c,
          ::std::complex< float > alpha = 1.0,
          ::std::complex< float > beta = 0.0, int lda = 1, int ldb = 1,
          int ldc = 1, Order order = Order::kRowMajor, Side side = Side::kLeft,
          Uplo Uplo = Uplo::kUpper);
void symm(int m, int n, const ::std::complex< double > *a,
          const ::std::complex< double > *b, ::std::complex< double > *c,
          ::std::complex< double > alpha = 1.0,
          ::std::complex< double > beta = 0.0, int lda = 1, int ldb = 1,
          int ldc = 1, Order order = Order::kRowMajor, Side side = Side::kLeft,
          Uplo Uplo = Uplo::kUpper);

void syrk(int n, int k, const float *a, const float *c, float alpha = 1.0,
          float beta = 0.0, int lda = 0, int ldc = 0,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper,
          Trans trans = Trans::kNoTrans);
void syrk(int n, int k, const double *a, const double *c, double alpha = 1.0,
          double beta = 0.0, int lda = 0, int ldc = 0,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper,
          Trans trans = Trans::kNoTrans);
void syrk(int n, int k, const ::std::complex< float > *a,
          const ::std::complex< float > *c, ::std::complex< float > alpha = 1.0,
          ::std::complex< float > beta = 0.0, int lda = 0, int ldc = 0,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper,
          Trans trans = Trans::kNoTrans);
void syrk(int n, int k, const ::std::complex< double > *a,
          const ::std::complex< double > *c,
          ::std::complex< double > alpha = 1.0,
          ::std::complex< double > beta = 0.0, int lda = 0, int ldc = 0,
          Order order = Order::kRowMajor, Uplo uplo = Uplo::kUpper,
          Trans trans = Trans::kNoTrans);

void syr2k(int n, int k, const float *a, const float *b, const float *c,
           float alpha = 1.0, float beta = 0.0, int lda = 0, int ldb = 0,
           int ldc = 0, Order order = Order::kRowMajor,
           Uplo uplo = Uplo::kUpper, Trans trans = Trans::kNoTrans);
void syr2k(int n, int k, const double *a, const float *b, const double *c,
           double alpha = 1.0, double beta = 0.0, int lda = 0, int ldb = 0,
           int ldc = 0, Order order = Order::kRowMajor,
           Uplo uplo = Uplo::kUpper, Trans trans = Trans::kNoTrans);
void syr2k(int n, int k, const ::std::complex< float > *a,
           const ::std::complex< float > *b, const ::std::complex< float > *c,
           ::std::complex< float > alpha = 1.0,
           ::std::complex< float > beta = 0.0, int lda = 0, int ldb = 0,
           int ldc = 0, Order order = Order::kRowMajor,
           Uplo uplo = Uplo::kUpper, Trans trans = Trans::kNoTrans);
void syr2k(int n, int k, const ::std::complex< double > *a,
           const ::std::complex< double > *b, const ::std::complex< double > *c,
           ::std::complex< double > alpha = 1.0,
           ::std::complex< double > beta = 0.0, int lda = 0, int ldb = 0,
           int ldc = 0, Order order = Order::kRowMajor,
           Uplo uplo = Uplo::kUpper, Trans trans = Trans::kNoTrans);

void trmm(int m, int n, const float *a, float *b, float alpha = 1.0,
          int lda = 0, int ldb = 0, Order order = Order::kRowMajor,
          Side side = Side::kLeft, Uplo uplo = Uplo::kUpper,
          Trans transa = Trans::kNoTrans, Diag diag = Diag::kNonUnit);
void trmm(int m, int n, const double *a, double *b, double alpha = 1.0,
          int lda = 0, int ldb = 0, Order order = Order::kRowMajor,
          Side side = Side::kLeft, Uplo uplo = Uplo::kUpper,
          Trans transa = Trans::kNoTrans, Diag diag = Diag::kNonUnit);
void trmm(int m, int n, const ::std::complex< float > *a,
          ::std::complex< float > *b, ::std::complex< float > alpha = 1.0,
          int lda = 0, int ldb = 0, Order order = Order::kRowMajor,
          Side side = Side::kLeft, Uplo uplo = Uplo::kUpper,
          Trans transa = Trans::kNoTrans, Diag diag = Diag::kNonUnit);
void trmm(int m, int n, const ::std::complex< double > *a,
          ::std::complex< double > *b, ::std::complex< double > alpha = 1.0,
          int lda = 0, int ldb = 0, Order order = Order::kRowMajor,
          Side side = Side::kLeft, Uplo uplo = Uplo::kUpper,
          Trans transa = Trans::kNoTrans, Diag diag = Diag::kNonUnit);

void trsm(int m, int n, const float *a, float *b, float alpha = 1.0,
          int lda = 0, int ldb = 0, Order order = Order::kRowMajor,
          Side side = Side::kLeft, Uplo uplo = Uplo::kUpper,
          Trans transa = Trans::kNoTrans, Diag diag = Diag::kNonUnit);
void trsm(int m, int n, const double *a, double *b, double alpha = 1.0,
          int lda = 0, int ldb = 0, Order order = Order::kRowMajor,
          Side side = Side::kLeft, Uplo uplo = Uplo::kUpper,
          Trans transa = Trans::kNoTrans, Diag diag = Diag::kNonUnit);
void trsm(int m, int n, const ::std::complex< float > *a,
          ::std::complex< float > *b, ::std::complex< float > alpha = 1.0,
          int lda = 0, int ldb = 0, Order order = Order::kRowMajor,
          Side side = Side::kLeft, Uplo uplo = Uplo::kUpper,
          Trans transa = Trans::kNoTrans, Diag diag = Diag::kNonUnit);
void trsm(int m, int n, const ::std::complex< double > *a,
          ::std::complex< double > *b, ::std::complex< double > alpha = 1.0,
          int lda = 0, int ldb = 0, Order order = Order::kRowMajor,
          Side side = Side::kLeft, Uplo uplo = Uplo::kUpper,
          Trans transa = Trans::kNoTrans, Diag diag = Diag::kNonUnit);

}  // namespace cxxblas
}  // namespace linalg
}  // namespace thunder

#endif  // THUNDER_LINALG_CXXBLAS_HPP_
