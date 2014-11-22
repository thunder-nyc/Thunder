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

#ifndef THUNDER_LINALG_BLAS_HPP_
#define THUNDER_LINALG_BLAS_HPP_

#include <complex>

extern "C" {

// BLAS level-1 functions

float sasum_(const int *n, const float *x, const int *incx);
float scasum_(const int *n, const ::std::complex< float > *x, const int *incx);
double dasum_(const int *n, const double *x, const int *incx);
double dzasum_(const int *n, const ::std::complex< double > *x,
               const int *incx);

void saxpy_(const int *n, const float *alpha, const float *x, const int *incx,
            float *y, const int *incy);
void daxpy_(const int *n, const double *alpha, const double *x, const int *incx,
            double *y, const int *incy);
void caxpy_(const int *n, const ::std::complex< float > *alpha,
            const ::std::complex< float > *x, const int *incx,
            ::std::complex< float > *y, const int *incy);
void zaxpy_(const int *n, const ::std::complex< double > *alpha,
            const ::std::complex< double > *x, const int *incx,
            ::std::complex< double > *y, const int *incy);

void scopy_(const int *n, const float *x, const int *incx, float *y,
            const int *incy);
void dcopy_(const int *n, const double *x, const int *incx, double *y,
            const int *incy);
void ccopy_(const int *n, const ::std::complex< float > *x, const int *incx,
            ::std::complex< float > *y, const int *incy);
void zcopy_(const int *n, const ::std::complex< double > *x, const int *incx,
            ::std::complex< double > *y, const int *incy);

float sdot_(const int *n, const float *x, const int *incx, const float *y,
             const int *incy);
double ddot_(const int *n, const double *x, const int *incx, const double *y,
             const int *incy);
float sdsdot_(const int *n, const float *sb, const float *sx, const int *incx,
              const float *sy, const int *incy);
double dsdot_(const int *n, const float *sx, const int *incx, const float *sy,
              const int *incy);
void cdotc_(::std::complex< float > *r, const int *n,
            const ::std::complex< float > *x, const int *incx,
            const ::std::complex< float > *y, const int *incy);
void zdotc_(::std::complex< double > *r, const int *n,
            const ::std::complex< double > *x, const int *incx,
            const ::std::complex< double > *y, const int *incy);
void cdotu_(::std::complex< float > *r, const int *n,
            const ::std::complex< float > *x, const int *incx,
            const ::std::complex< float > *y, const int *incy);
void zdotu_(::std::complex< double > *r, const int *n,
            const ::std::complex< double > *x, const int *incx,
            const ::std::complex< double > *y, const int *incy);

float snrm2_(const int *n, const float *x, const int *incx);
double dnrm2_(const int *n, const double *x, const int *incx);
float scnrm2_(const int *n, const ::std::complex< float > *x, const int *incx);
double dznrm2_(const int *n, const ::std::complex< double > *x,
               const int *incx);

void srot_(const int *n, float *x, const int *incx, float *y, const int *incy,
           const float *c, const float *s);
void drot_(const int *n, double *x, const int *incx, double *y, const int *incy,
           const double *c, const double *s);
void csrot_(const int *n, ::std::complex< float > *x, const int *incx,
            ::std::complex< float > *y, const int *incy, const float *c,
            const float *s);
void zdrot_(const int *n, ::std::complex< double > *x, const int *incx,
            ::std::complex< double > *y, const int *incy, const double *c,
            const double *s);

void srotg_(float *a, float *b, float *c, float *s);
void drotg_(double *a, double *b, double *c, double *s);
void crotg_(::std::complex< float > *a, ::std::complex< float > *b,
            float *c, ::std::complex< float > *s);
void zrotg_(::std::complex< double > *a, ::std::complex< double > *b,
            double *c, ::std::complex< double > *s);

void srotm_(const int *n, float *x, const int *incx, float *y, const int *incy,
            const float *param);
void drotm_(const int *n, double *x, const int *incx, double *y,
            const int *incy, const double *param);

void srotmg_(float *d1, float *d2, float *x1, const float *x2,
             const float *param);
void drotmg_(double *d1, double *d2, float *x1, const double *x2,
             const double *param);


void sscal_(const int *n, const float *a, float *x, const int *incx);
void dscal_(const int *n, const double *a, double *x, const int *incx);
void cscal_(const int *n, const ::std::complex< float > *a,
            ::std::complex< float > *x, const int *incx);
void zscal_(const int *n, const ::std::complex< double > *a,
            ::std::complex< double > *x, const int *incx);
void csscal_(const int *n, const float *a, ::std::complex< float > *x,
             const int *incx);
void zdscal_(const int *n, const double *a, ::std::complex< double > *x,
             const int *incx);

void sswap_(const int *n, float *x, const int *incx, float *y,
            const int *incy);
void dswap_(const int *n, double *x, const int *incx, double *y,
            const int *incy);
void cswap_(const int *n, ::std::complex< float > *x, const int *incx,
            ::std::complex< float > *y, const int *incy);
void zswap_(const int *n, ::std::complex< double > *x, const int *incx,
            ::std::complex< double > *y, const int *incy);

int isamax_(const int *n, const float *x, const int *incx);
int idamax_(const int *n, const double *x, const int *incx);
int icamax_(const int *n, const ::std::complex< float > *x, const int *incx);
int izamax_(const int *n, const ::std::complex< double > *x, const int *incx);


// BLAS level-2 routines -- note that these are column major!

void sgbmv_(const char *trans, const int *m, const int *n, const int *kl,
            const int *ku, const float *alpha, const float *a, const int *lda,
            const float *x, const int *incx, const float *beta, float *y,
            const int *incy);
void dgbmv_(const char *trans, const int *m, const int *n, const int *kl,
            const int *ku, const double *alpha, const double *a, const int *lda,
            const double *x, const int *incx, const double *beta, double *y,
            const int *incy);
void cgbmv_(const char *trans, const int *m, const int *n, const int *kl,
            const int *ku, const ::std::complex< float > *alpha,
            const ::std::complex< float > *a, const int *lda,
            const ::std::complex< float > *x, const int *incx,
            const ::std::complex< float > *beta, ::std::complex< float > *y,
            const int *incy);
void zgbmv_(const char *trans, const int *m, const int *n, const int *kl,
            const int *ku, const ::std::complex< double > *alpha,
            const ::std::complex< double > *a, const int *lda,
            const ::std::complex< double > *x, const int *incx,
            const ::std::complex< double > *beta, ::std::complex< double > *y,
            const int *incy);

void sgemv_(const char *trans, const int *m, const int *n, const float *alpha,
            const float *a, const int *lda, const float *x, const int *incx,
            const float *beta, float *y, const int *incy);
void dgemv_(const char *trans, const int *m, const int *n, const double *alpha,
            const double *a, const int *lda, const double *x, const int *incx,
            const double *beta, double *y, const int *incy);
void cgemv_(const char *trans, const int *m, const int *n,
            const ::std::complex< float > *alpha,
            const ::std::complex< float > *a, const int *lda,
            const ::std::complex< float > *x, const int *incx,
            const ::std::complex< float > *beta, ::std::complex< float > *y,
            const int *incy);
void zgemv_(const char *trans, const int *m, const int *n,
            const ::std::complex< double > *alpha,
            const ::std::complex< double > *a, const int *lda,
            const ::std::complex< double > *x, const int *incx,
            const ::std::complex< double > *beta, ::std::complex< double > *y,
            const int *incy);
void scgemv_(const char *trans, const int *m, const int *n,
             const ::std::complex< float > *alpha,
             const float *a, const int *lda, const ::std::complex< float > *x,
             const int *incx, const ::std::complex< float > *beta,
             ::std::complex< double > *y, const int *incy);
void dzgemv_(const char *trans, const int *m, const int *n,
             const ::std::complex< double > *alpha,
             const double *a, const int *lda,
             const ::std::complex< double > *x, const int *incx,
             const ::std::complex< double > *beta, ::std::complex< double > *y,
             const int *incy);

void sger_(const int *m, const int *n, const float *alpha, const float *x,
           const int *incx, const float *y, const int *incy, float *a,
           const int *lda);
void dger_(const int *m, const int *n, const double *alpha, const double *x,
           const int *incx, const double *y, const int *incy, double *a,
           const int *lda);

void cgerc_(const int *m, const int *n, const ::std::complex< float > *alpha,
            const ::std::complex< float > *x, const int *incx,
            const ::std::complex< float > *y, const int *incy,
            ::std::complex< float > *a, const int *lda);
void zgerc_(const int *m, const int *n, const ::std::complex< double > *alpha,
            const ::std::complex< double > *x, const int *incx,
            const ::std::complex< double > *y, const int *incy,
            ::std::complex< double > *a, const int *lda);

void cgeru_(const int *m, const int *n, const ::std::complex< float > *alpha,
            const ::std::complex< float > *x, const int *incx,
            const ::std::complex< float > *y, const int *incy,
            ::std::complex< float > *a, const int *lda);
void zgeru_(const int *m, const int *n, const ::std::complex< double > *alpha,
            const ::std::complex< double > *x, const int *incx,
            const ::std::complex< double > *y, const int *incy,
            ::std::complex< double > *a, const int *lda);

void chbmv_(const char *uplo, const int *n, const int *k,
            const ::std::complex< float > *alpha,
            const ::std::complex< float > *a, const int *lda,
            const ::std::complex< float > *x, const int *incx,
            const ::std::complex< float > *beta, ::std::complex< double > *y,
            const int *incy);
void zhbmv_(const char *uplo, const int *n, const int *k,
            const ::std::complex< double > *alpha,
            const ::std::complex< double > *a, const int *lda,
            const ::std::complex< double > *x, const int *incx,
            const ::std::complex< double > *beta, ::std::complex< double > *y,
            const int *incy);

void chemv_(const char *uplo, const int *n,
            const ::std::complex< float > *alpha,
            const ::std::complex< float > *a, const int *lda,
            const ::std::complex< float > *x, const int *incx,
            const ::std::complex< float > *beta, ::std::complex< double > *y,
            const int *incy);
void zhemv_(const char *uplo, const int *n,
            const ::std::complex< double > *alpha,
            const ::std::complex< double > *a, const int *lda,
            const ::std::complex< double > *x, const int *incx,
            const ::std::complex< double > *beta, ::std::complex< double > *y,
            const int *incy);

void cher_(const char *uplo, const int *n, const ::std::complex< float > *alpha,
           const ::std::complex< float > *x, const int *incx,
           ::std::complex< float > *a, const int *lda);
void zher_(const char *uplo, const int *n,
           const ::std::complex< double > *alpha,
           const ::std::complex< double > *x, const int *incx,
           ::std::complex< double > *a, const int *lda);

void cher2_(const char *uplo, const int *n,
            const ::std::complex< float > *alpha,
            const ::std::complex< float > *x, const int *incx,
            const ::std::complex< float > *y, const int *incy,
            ::std::complex< float > *a, const int *lda);
void zher2_(const char *uplo, const int *n,
            const ::std::complex< double > *alpha,
            const ::std::complex< double > *x, const int *incx,
            const ::std::complex< double > *y, const int *incy,
            ::std::complex< double > *a, const int *lda);

void chpmv_(const char *uplo, const int *n,
            const ::std::complex< float > *alpha,
            const ::std::complex< float > *ap,
            const ::std::complex< float > *x, const int *incx,
            const ::std::complex< float > *beta, ::std::complex< double > *y,
            const int *incy);
void zhpmv_(const char *uplo, const int *n,
            const ::std::complex< double > *alpha,
            const ::std::complex< double > *ap,
            const ::std::complex< double > *x, const int *incx,
            const ::std::complex< double > *beta, ::std::complex< double > *y,
            const int *incy);

void chpr_(const char *uplo, const int *n, const ::std::complex< float > *alpha,
           const ::std::complex< float > *x, const int *incx,
           ::std::complex< float > *ap);
void zhpr_(const char *uplo, const int *n,
           const ::std::complex< double > *alpha,
           const ::std::complex< double > *x, const int *incx,
           ::std::complex< double > *ap);

void chpr2_(const char *uplo, const int *n,
            const ::std::complex< float > *alpha,
            const ::std::complex< float > *x, const int *incx,
            const ::std::complex< float > *y, const int *incy,
            ::std::complex< float > *ap);
void zhpr2_(const char *uplo, const int *n,
            const ::std::complex< double > *alpha,
            const ::std::complex< double > *x, const int *incx,
            const ::std::complex< double > *y, const int *incy,
            ::std::complex< double > *ap);

void ssbmv_(const char *uplo, const int *n, const int *k, const float *alpha,
            const float *a, const int *lda, const float *x, const int *incx,
            const float *beta, float *y, const int *incy);
void dsbmv_(const char *uplo, const int *n, const int *k, const double *alpha,
            const double *a, const int *lda, const double *x, const int *incx,
            const double *beta, double *y, const int *incy);

void sspmv_(const char *uplo, const int *n, const float *alpha, const float *ap,
            const float *x, const int *incx, const float *beta, float *y,
            const int *incy);
void dspmv_(const char *uplo, const int *n, const double *alpha,
            const double *ap, const double *x, const int *incx,
            const double *beta, double *y, const int *incy);

void sspr_(const char *uplo, const int *n, const float *alpha,
           const float *x, const int *incx, float *ap);
void dspr_(const char *uplo, const int *n, const double *alpha,
           const double *x, const int *incx, double *ap);

void sspr2_(const char *uplo, const int *n, const float *alpha,
            const float *x, const int *incx, const float *y, const int *incy,
            float *ap);
void dspr2_(const char *uplo, const int *n, const double *alpha,
            const double *x, const int *incx, const double *y, const int *incy,
            double *ap);

void ssymv_(const char *uplo, const int *n, const float *alpha, const int *lda,
            const float *x, const int *incx, const float *beta, float *y,
            const int *incy);
void dsymv_(const char *uplo, const int *n, const double *alpha, const int *lda,
            const double *x, const int *incx, const double *beta,
            float *y, const int *incy);

void ssyr_(const char *uplo, const int *n, const float *alpha,
           const float *x, const int *incx, float *a, const int *lda);
void dsyr_(const char *uplo, const int *n, const double *alpha,
           const double *x, const int *incx, double *a, const int *lda);

void ssyr2_(const char *uplo, const int *n, const float *alpha,
            const float *x, const int *incx, const float *y, const int *incy,
            float *a, const int *lda);
void dsyr2_(const char *uplo, const int *n, const double *alpha,
            const double *x, const int *incx, const double *y, const int *incy,
            double *a, const int *lda);

void stbmv_(const char *uplo, const char *trans, const char *diag, const int *n,
            const int *k, const float *a, const int *lda, float *x,
            const int *incx);
void dtbmv_(const char *uplo, const char *trans, const char *diag, const int *n,
            const int *k, const double *a, const int *lda, double *x,
            const int *incx);
void ctbmv_(const char *uplo, const char *trans, const char *diag, const int *n,
            const int *k, const ::std::complex< float > *a, const int *lda,
            ::std::complex< float > *x, const int *incx);
void ztbmv_(const char *uplo, const char *trans, const char *diag, const int *n,
            const int *k, const ::std::complex< double > *a, const int *lda,
            ::std::complex< double > *x, const int *incx);

void stbsv_(const char *uplo, const char *trans, const char *diag, const int *n,
            const int *k, const float *a, const int *lda, float *x,
            const int *incx);
void dtbsv_(const char *uplo, const char *trans, const char *diag, const int *n,
            const int *k, const double *a, const int *lda, double *x,
            const int *incx);
void ctbsv_(const char *uplo, const char *trans, const char *diag, const int *n,
            const int *k, const ::std::complex< float > *a, const int *lda,
            ::std::complex< float > *x, const int *incx);
void ztbsv_(const char *uplo, const char *trans, const char *diag, const int *n,
            const int *k, const ::std::complex< double > *a, const int *lda,
            ::std::complex< double > *x, const int *incx);

void stpmv_(const char *uplo, const char *trans, const char *diag, const int *n,
            const float *ap, float *x, const int *incx);
void dtpmv_(const char *uplo, const char *trans, const char *diag, const int *n,
            const double *ap, double *x, const int *incx);
void ctpmv_(const char *uplo, const char *trans, const char *diag, const int *n,
            const ::std::complex< float > *ap, ::std::complex< float > *x,
            const int *incx);
void ztpmv_(const char *uplo, const char *trans, const char *diag, const int *n,
            const ::std::complex< double > *ap,
            ::std::complex< double > *x, const int *incx);

void stpsv_(const char *uplo, const char *trans, const char *diag, const int *n,
            const float *ap, float *x, const int *incx);
void dtpsv_(const char *uplo, const char *trans, const char *diag, const int *n,
            const double *ap, double *x, const int *incx);
void ctpsv_(const char *uplo, const char *trans, const char *diag, const int *n,
            const ::std::complex< float > *ap, ::std::complex< float > *x,
            const int *incx);
void ztpsv_(const char *uplo, const char *trans, const char *diag, const int *n,
            const ::std::complex< double > *ap, ::std::complex< double > *x,
            const int *incx);

void strmv_(const char *uplo, const char *trans, const char *diag, const int *n,
            const float *a, const int *lda, float *x, const int *incx);
void dtrmv_(const char *uplo, const char *trans, const char *diag, const int *n,
            const double *a, const int *lda, double *x, const int *incx);
void ctrmv_(const char *uplo, const char *trans, const char *diag, const int *n,
            const ::std::complex< float > *a, const int *lda,
            ::std::complex< float > *x, const int *incx);
void ztrmv_(const char *uplo, const char *trans, const char *diag, const int *n,
            const ::std::complex< double > *a, const int *lda,
            ::std::complex< double > *x, const int *incx);

void strsv_(const char *uplo, const char *trans, const char *diag, const int *n,
            const float *a, const int *lda, float *x, const int *incx);
void dtrsv_(const char *uplo, const char *trans, const char *diag, const int *n,
            const double *a, const int *lda, double *x, const int *incx);
void ctrsv_(const char *uplo, const char *trans, const char *diag, const int *n,
            const ::std::complex< float > *a, const int *lda,
            ::std::complex< float > *x, const int *incx);
void ztrsv_(const char *uplo, const char *trans, const char *diag, const int *n,
            const ::std::complex< double > *a, const int *lda,
            ::std::complex< double > *x, const int *incx);


// BLAS level-3 routines -- note that these functions are column-major!

void sgemm_(const char *transa, const char *transb, const int *m, const int *n,
            const int *k, const float *alpha, const float *a, const int *lda,
            const float *b, const int *ldb, const float *beta, float *c,
            const int *ldc);
void dgemm_(const char *transa, const char *transb, const int *m, const int *n,
            const int *k, const double *alpha, const double *a, const int *lda,
            const double *b, const int *ldb, const double *beta, double *c,
            const int *ldc);
void cgemm_(const char *transa, const char *transb, const int *m, const int *n,
            const int *k, const ::std::complex< float > *alpha,
            const ::std::complex< float > *a, const int *lda,
            const ::std::complex< float > *b, const int *ldb,
            const ::std::complex< float > *beta,
            ::std::complex< float > *c, const int *ldc);
void zgemm_(const char *transa, const char *transb, const int *m, const int *n,
            const int *k, const ::std::complex< double > *alpha,
            const ::std::complex< double > *a, const int *lda,
            const ::std::complex< double > *b, const int *ldb,
            const ::std::complex< double > *beta,
            ::std::complex< double > *c, const int *ldc);
void scgemm_(const char *transa, const char *transb, const int *m, const int *n,
             const int *k, const ::std::complex< float > *alpha,
             const float *a, const int *lda, const ::std::complex< float > *b,
             const int *ldb, const ::std::complex< float > *beta,
             ::std::complex< float > *c, const int *ldc);
void dzgemm_(const char *transa, const char *transb, const int *m, const int *n,
             const int *k, const ::std::complex< double > *alpha,
             const double *a, const int *lda, const ::std::complex< double > *b,
             const int *ldb, const ::std::complex< double > *beta,
             ::std::complex< double > *c, const int *ldc);

void chemm_(const char *side, const char *uplo, const int *m, const int *n,
            const ::std::complex< float > *alpha,
            const ::std::complex< float > *a, const int *lda,
            const ::std::complex< float > *b, const int *ldb,
            const ::std::complex< float > *beta,
            ::std::complex< float > *c, const int *ldc);
void zhemm_(const char *side, const char *uplo, const int *m, const int *n,
            const ::std::complex< double > *alpha,
            const ::std::complex< double > *a, const int *lda,
            const ::std::complex< double > *b, const int *ldb,
            const ::std::complex< double > *beta,
            ::std::complex< double > *c, const int *ldc);

void cherk_(const char *uplo, const char *trans, const int *n, const int *k,
            const ::std::complex< float > *alpha,
            const ::std::complex< float > *a, const int *lda,
            const ::std::complex< float > *beta,
            const ::std::complex< float > *c, const int *ldc);
void zherk_(const char *uplo, const char *trans, const int *n, const int *k,
            const ::std::complex< double > *alpha,
            const ::std::complex< double > *a, const int *lda,
            const ::std::complex< double > *beta,
            const ::std::complex< double > *c, const int *ldc);
void cher2k_(const char *uplo, const char *trans, const int *n, const int *k,
             const ::std::complex< float > *alpha,
             const ::std::complex< float > *a, const int *lda,
             const ::std::complex< float > *b, const int *ldb,
             const ::std::complex< float > *beta,
             const ::std::complex< float > *c, const int *ldc);
void zher2k_(const char *uplo, const char *trans, const int *n, const int *k,
             const ::std::complex< double > *alpha,
             const ::std::complex< double > *a, const int *lda,
             const ::std::complex< double > *b, const int *ldb,
             const ::std::complex< double > *beta,
             const ::std::complex< double > *c, const int *ldc);

void ssymm_(const char *side, const char *uplo, const int *m, const int *n,
            const float *alpha, const float *a, const int *lda, const float *b,
            const int *ldb, const float *beta, float *c, const int *ldc);
void dsymm_(const char *side, const char *uplo, const int *m, const int *n,
            const double *alpha, const double *a, const int *lda,
            const double *b, const int *ldb, const double *beta, double *c,
            const int *ldc);
void csymm_(const char *side, const char *uplo, const int *m, const int *n,
            const ::std::complex< float > *alpha,
            const ::std::complex< float > *a, const int *lda,
            const ::std::complex< float > *b, const int *ldb,
            const ::std::complex< float > *beta,
            ::std::complex< float > *c, const int *ldc);
void zsymm_(const char *side, const char *uplo, const int *m, const int *n,
            const ::std::complex< double > *alpha,
            const ::std::complex< double > *a, const int *lda,
            const ::std::complex< double > *b, const int *ldb,
            const ::std::complex< double > *beta,
            ::std::complex< double > *c, const int *ldc);

void ssyrk_(const char *uplo, const char *trans, const int *n, const int *k,
            const float *alpha, const float *a, const int *lda,
            const float *beta, float *c, const int *ldc);
void dsyrk_(const char *uplo, const char *trans, const int *n, const int *k,
            const double *alpha, const double *a, const int *lda,
            const double *beta, double *c, const int *ldc);
void csyrk_(const char *uplo, const char *trans, const int *n, const int *k,
            const ::std::complex< float > *alpha,
            const ::std::complex< float > *a, const int *lda,
            const ::std::complex< float > *beta,
            ::std::complex< float > *c, const int *ldc);
void zsyrk_(const char *uplo, const char *trans, const int *n, const int *k,
            const ::std::complex< double > *alpha,
            const ::std::complex< double > *a, const int *lda,
            const ::std::complex< double > *beta,
            ::std::complex< double > *c, const int *ldc);

void ssyr2k_(const char *uplo, const char *trans, const int *n, const int *k,
             const float *alpha, const float *a, const int *lda, const float *b,
             const int *ldb, const float *beta, float *c, const int *ldc);
void dsyr2k_(const char *uplo, const char *trans, const int *n, const int *k,
             const double *alpha, const double *a, const int *lda,
             const double *b, const int *ldb, const double *beta, double *c,
             const int *ldc);
void csyr2k_(const char *uplo, const char *trans, const int *n, const int *k,
             const ::std::complex< float > *alpha,
             const ::std::complex< float > *a, const int *lda,
             const ::std::complex< float > *b, const int *ldb,
             const ::std::complex< float > *beta,
             ::std::complex< float > *c, const int *ldc);
void zsyr2k_(const char *uplo, const char *trans, const int *n, const int *k,
             const ::std::complex< double > *alpha,
             const ::std::complex< double > *a, const int *lda,
             const ::std::complex< double > *b, const int *ldb,
             const ::std::complex< double > *beta,
             ::std::complex< double > *c, const int *ldc);

void strmm_(const char *side, const char *uplo, const char *transa,
            const char *diag, const int *m, const int *n, const float *alpha,
            const float *a, const int *lda, float *b, const int *ldb);
void dtrmm_(const char *side, const char *uplo, const char *transa,
            const char *diag, const int *m, const int *n, const double *alpha,
            const double *a, const int *lda, double *b, const int *ldb);
void ctrmm_(const char *side, const char *uplo, const char *transa,
            const char *diag, const int *m, const int *n,
            const ::std::complex< float > *alpha,
            const ::std::complex< float > *a, const int *lda,
            ::std::complex< float > *b, const int *ldb);
void ztrmm_(const char *side, const char *uplo, const char *transa,
            const char *diag, const int *m, const int *n,
            const ::std::complex< double > *alpha,
            const ::std::complex< double > *a, const int *lda,
            ::std::complex< double > *b, const int *ldb);

void strsm_(const char *side, const char *uplo, const char *transa,
            const char *diag, const int *m, const int *n, const float *alpha,
            const float *a, const int *lda, float *b, const int *ldb);
void dtrsm_(const char *side, const char *uplo, const char *transa,
            const char *diag, const int *m, const int *n, const double *alpha,
            const double *a, const int *lda, double *b, const int *ldb);
void ctrsm_(const char *side, const char *uplo, const char *transa,
            const char *diag, const int *m, const int *n,
            const ::std::complex< float > *alpha,
            const ::std::complex< float > *a, const int *lda,
            ::std::complex< float > *b, const int *ldb);
void ztrsm_(const char *side, const char *uplo, const char *transa,
            const char *diag, const int *m, const int *n,
            const ::std::complex< double > *alpha,
            const ::std::complex< double > *a, const int *lda,
            ::std::complex< double > *b, const int *ldb);

}  // extern "C"

#endif  // THUNDER_LINALG_BLAS_HPP_
