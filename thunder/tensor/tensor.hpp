/*
 * Copyright 2014 Xiang Zhang. All Rights Reserved.
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
 */

#ifndef THUNDER_TENSOR_TENSOR_HPP
#define THUNDER_TENSOR_TENSOR_HPP

#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <utility>

#include "thunder/storage.hpp"

namespace thunder {

template < typename S >
class Tensor {
 public:
  // Typedefs from storage
  typedef typename S::allocator_type allocator_type;
  typedef typename S::value_type value_type;
  typedef typename S::reference reference;
  typedef typename S::const_reference const_reference;
  typedef typename S::difference_type difference_type;
  typedef typename S::size_type size_type;
  typedef typename S::pointer pointer;
  typedef typename S::const_pointer const_pointer;

  // Typedefs for tensor
  typedef Storage< size_type > size_storage;
  typedef Storage< difference_type > stride_storage;
  typedef ::std::shared_ptr< S > storage_pointer;
  typedef typename size_storage::size_type dim_type;

  // Constructors
  explicit Tensor(const storage_pointer &storage = storage_pointer(new S()),
                  size_type offset = 0);
  explicit Tensor(const size_storage &size);
  Tensor(const size_storage &size, const storage_pointer &storage,
         size_type offset = 0);
  Tensor(const size_storage &size, const stride_storage &stride);
  Tensor(const size_storage &size, const stride_storage &stride,
         const storage_pointer &storage, size_type offset = 0);
  Tensor(const Tensor &other);
  Tensor(Tensor &&other);

  // Destructor
  virtual ~Tensor();

  // Assignment operators
  virtual Tensor& operator=(Tensor other);
  virtual const Tensor &operator=(const_reference value) const;
  virtual Tensor& operator=(const_reference value);

  // Static casts
  operator value_type() const;

  // Paranthesis operators points to a reference of value
  virtual reference operator()() const;
  virtual reference operator()(size_type pos) const;
  virtual reference operator()(size_type pos0, size_type pos1) const;
  virtual reference operator()(size_type pos0, size_type pos1,
                               size_type pos2) const;
  virtual reference operator()(size_type pos0, size_type pos1,
                               size_type pos2, size_type pos_3) const;
  virtual reference operator()(const size_storage &pos) const;

  // Index operators
  virtual Tensor operator[](size_type pos) const;
  virtual Tensor operator[](
      const ::std::pair< size_type, size_type > &range) const;

  // Arithmetic operators with value are delegated
  virtual Tensor operator+(const_reference value) const;
  virtual Tensor operator-(const_reference value) const;
  template < typename Other_S >
  friend Tensor< Other_S > operator+(
      typename Tensor< Other_S >::const_reference value, const Tensor< S > &t);
  template < typename Other_S >
  friend Tensor< Other_S > operator-(
      typename Tensor< Other_S >::const_reference value, const Tensor< S > &t);

  // Templated airthmetic operators are type left associated
  template < typename Other_S >
  Tensor operator+(const Tensor< Other_S > &other) const;
  template < typename Other_S >
  Tensor operator-(const Tensor< Other_S > &other) const;

  // Arithmetic operators are delegated
  virtual Tensor operator+(const Tensor &other) const;
  virtual Tensor operator-(const Tensor &other) const;

  // Comparison operators with value are delegated
  virtual Tensor operator==(const_reference value) const;
  virtual Tensor operator>(const_reference value) const;
  virtual Tensor operator<(const_reference value) const;
  virtual Tensor operator<=(const_reference value) const;
  virtual Tensor operator>=(const_reference value) const;
  template < typename Other_S >
  friend Tensor< Other_S > operator==(
      typename Tensor< Other_S >::const_reference value, const Tensor< S > &t);
  template < typename Other_S >
  friend Tensor< Other_S > operator>(
      typename Tensor< Other_S >::const_reference value, const Tensor< S > &t);
  template < typename Other_S >
  friend Tensor< Other_S > operator<(
      typename Tensor< Other_S >::const_reference value, const Tensor< S > &t);
  template < typename Other_S >
  friend Tensor< Other_S > operator>=(
      typename Tensor< Other_S >::const_reference value, const Tensor< S > &t);
  template < typename Other_S >
  friend Tensor< Other_S > operator<=(
      typename Tensor< Other_S >::const_reference value, const Tensor< S > &t);

  // Templated comparison operators are delegated
  template < typename Other_S >
  Tensor operator==(const Tensor< Other_S > &other);
  template < typename Other_S >
  Tensor operator>(const Tensor< Other_S > &other);
  template < typename Other_S >
  Tensor operator<(const Tensor< Other_S > &other);
  template < typename Other_S >
  Tensor operator<=(const Tensor< Other_S > &other);
  template < typename Other_S >
  Tensor operator>=(const Tensor< Other_S > &other);

  // Comparison operators with another tensor are delegated
  virtual Tensor operator==(const Tensor &other) const;
  virtual Tensor operator>(const Tensor &other) const;
  virtual Tensor operator<(const Tensor &other) const;
  virtual Tensor operator<=(const Tensor &other) const;
  virtual Tensor operator>=(const Tensor &other) const;

  // Iterators and their functions. Subtensor and value iterators.
  class iterator;
  class reference_iterator;
  virtual iterator begin() const;
  virtual iterator end() const;
  virtual reference_iterator value_begin() const;
  virtual reference_iterator value_end() const;

  // Static iterator functions are delegated
  static iterator begin(const Tensor &t);
  static iterator end(const Tensor &t);
  static reference_iterator value_begin(const Tensor &t);
  static reference_iterator value_end(const Tensor &t);

  // Non-virtual templated queriers
  template < typename Other_S >
  bool IsSameSizeAs(const Tensor< Other_S > &other) const;

  // Static non-virtual templated queriers are delegated
  template < typename Other_S >
  bool IsSameSizeAs(const Tensor& t, const Tensor< Other_S > &other) const;

  // Non-virtual templated modifiers
  template < typename Other_S >
  Tensor& Copy(const Tensor< Other_S > &other);
  template < typename Other_S >
  Tensor& ResizeAs(const Tensor< Other_S > &other);

  // Static non-virtual templated modifiers are delegated
  template < typename Other_S >
  Tensor& Copy(Tensor *t, const Tensor< Other_S > &other);
  template < typename Other_S >
  Tensor& ResizeAs(Tensor *t, const Tensor< Other_S > &other);

  // Property queries
  virtual dim_type Dimension() const;
  virtual size_storage Size() const;
  virtual size_type Size(dim_type dim) const;
  virtual size_type Count() const;
  virtual stride_storage Stride() const;
  virtual difference_type Stride(dim_type dim) const;
  virtual storage_pointer GetStorage() const;
  virtual size_type Offset() const;
  virtual pointer Data() const;
  virtual bool IsContiguous() const;
  virtual bool IsSameSizeAs(const Tensor &other) const;

  // Static property queries are delegated
  static dim_type Dimension(const Tensor &t);
  static size_storage Size(const Tensor &t);
  static size_type Size(const Tensor &t, dim_type dim);
  static size_type Count(const Tensor &t);
  static stride_storage Stride(const Tensor &t);
  static difference_type Stride(const Tensor &t, dim_type dim);
  static storage_pointer GetStorage(const Tensor &t);
  static size_type Offset(const Tensor &t);
  static pointer Data(const Tensor &t);
  static bool IsContiguous(const Tensor &t);
  static bool IsSameSizeAs(const Tensor &t, const Tensor &other);

  // Normal and specialized modifiers
  virtual Tensor& Copy(const Tensor &other);
  virtual Tensor& Set(const Tensor &other);
  virtual Tensor& Set(const storage_pointer &storage, size_type offset = 0);
  virtual Tensor& Set(const size_storage &size, const storage_pointer &storage,
                      size_type offset = 0);
  virtual Tensor& Set(const size_storage &size, const stride_storage &stride,
                      const storage_pointer &storage, size_type offset = 0);
  virtual Tensor& Resize(const size_storage &size);
  virtual Tensor& Resize(const size_storage &size,
                         const stride_storage &stride);
  virtual Tensor& ResizeAs(const Tensor &other);
  virtual Tensor& Contiguous();

  // Static normal and specialized modifiers are delegated
  static Tensor& Copy(Tensor *t, const Tensor &other);
  static Tensor& Set(Tensor *t, const Tensor &other);
  static Tensor& Set(Tensor *t, const storage_pointer &storage,
                     size_type offset = 0);
  static Tensor& Set(Tensor *t, const size_storage &size,
                     const storage_pointer &storage, size_type offset = 0);
  static Tensor& Set(Tensor *t, const size_storage &size,
                     const stride_storage &stride,
                     const storage_pointer &storage, size_type offset = 0);
  static Tensor& Resize(Tensor *t, const size_storage &size);
  static Tensor& Resize(Tensor *t, const size_storage &size,
                        const stride_storage &stride);
  static Tensor& ResizeAs(Tensor *t, const Tensor &other);
  static Tensor& Contiguous(Tensor *t);

  // Templated subtensor extractors
  template < typename Other_S >
  Tensor ViewAs(const Tensor< Other_S > &other) const;

  // Static templated subtensor extractors are delegated
  template < typename Other_S >
  static Tensor ViewAs(const Tensor& t, const Tensor< Other_S > &other);

  // Extract subtensors or transformations
  virtual Tensor Narrow(dim_type dim, size_type pos, size_type size) const;
  virtual Tensor Select(dim_type dim, size_type pos) const;
  virtual Tensor View(const size_storage &size) const;
  virtual Tensor ViewAs(const Tensor &other) const;
  virtual Tensor Transpose() const;
  virtual Tensor Transpose(dim_type dim0, dim_type dim1) const;
  virtual Tensor Unfold(dim_type dim, size_type size, size_type step) const;
  virtual Tensor Clone() const;

  // Static subtensor or transformation extractors are delegated
  static Tensor Narrow(const Tensor &t, dim_type dim, size_type pos,
                       size_type size);
  static Tensor Select(const Tensor &t, dim_type dim, size_type pos);
  static Tensor View(const Tensor &t, const size_storage &size);
  static Tensor ViewAs(const Tensor &t, const Tensor &other);
  static Tensor Transpose(const Tensor &t);
  static Tensor Transpose(const Tensor &t, dim_type dim0, dim_type dim1);
  static Tensor Unfold(const Tensor &t, dim_type dim, size_type size,
                       size_type step);
  static Tensor Clone(const Tensor& t);

  // lambda applications
  virtual const Tensor& Apply(
      const ::std::function< value_type(value_type) > &lambda) const;
  virtual const Tensor& Apply(
      const ::std::function< value_type(const value_type&) > &lambda) const;
  virtual const Tensor& Apply(
      const ::std::function< void(value_type&) > &lambda) const;
  virtual const Tensor& Apply(
      const ::std::function< void(value_type*) > &lambda) const;

  // Non-const lambda applications are delegated using const_cast
  virtual Tensor& Apply(
      const ::std::function< value_type(value_type) > &lambda);
  virtual Tensor& Apply(
      const ::std::function< value_type(const value_type&) > &lambda);
  virtual Tensor& Apply(const ::std::function< void(value_type&) > &lambda);
  virtual Tensor& Apply(const ::std::function< void(value_type*) > &lambda);

  // Static lambda applications are delegated
  static Tensor Apply(const Tensor& t,
                      const ::std::function< value_type(value_type) > &lambda);
  static Tensor Apply(
      const Tensor& t,
      const ::std::function< value_type(const value_type&) > &lambda);
  static Tensor Apply(const Tensor& t,
                      const ::std::function< void(value_type&) > &lambda);
  static Tensor Apply(const Tensor& t,
                      const ::std::function< void(value_type*) > &lambda);

  // Reduction operations
  virtual value_type Max() const;
  virtual value_type Min() const;
  virtual value_type Sum() const;
  virtual value_type Mean() const;
  virtual value_type Var() const;
  virtual value_type Std() const;

  // Static reduction operations are deligated
  static value_type Max(const Tensor &t);
  static value_type Min(const Tensor &t);
  static value_type Sum(const Tensor &t);
  static value_type Mean(const Tensor &t);
  static value_type Var(const Tensor &t);
  static value_type Std(const Tensor &t);

  // All random generators are either double or int, using static_cast.
  virtual const Tensor& Rand() const;
  virtual const Tensor& Uniform(double a = 0.0, double b = 1.0) const;
  virtual const Tensor& UniformReal(double a = 0.0, double b = 1.0) const;
  virtual const Tensor& UniformInt(
      int a = 0, int b = ::std::numeric_limits< int >::max()) const;
  virtual const Tensor& Bernoulli(double p = 0.5) const;
  virtual const Tensor& Binomial(int t = 1, double p = 0.5) const;
  virtual const Tensor& NegativeBinomial(int k = 1, double p = 0.5) const;
  virtual const Tensor& Geometric(double p = 0.5) const;
  virtual const Tensor& Poisson(double mean = 1.0) const;
  virtual const Tensor& Exponential(double lambda = 1.0) const;
  virtual const Tensor& Gamma(double alpha = 1.0, double beta = 1.0) const;
  virtual const Tensor& Weibull(double a = 1.0, double b = 1.0) const;
  virtual const Tensor& ExtremeValue(double a = 0.0, double b = 1.0) const;
  virtual const Tensor& Normal(double mean = 0.0, double stddev = 1.0) const;
  virtual const Tensor& LogNormal(double m = 0.0, double s = 1.0) const;
  virtual const Tensor& ChiSquared(double n = 1.0) const;
  virtual const Tensor& Cauchy(double a = 0.0, double b = 1.0) const;
  virtual const Tensor& FisherF(double m = 1.0, double n = 1.0) const;
  virtual const Tensor& StudentT(double n = 1.0) const;

  // All non-const random generators are delegated using const_cast.
  virtual Tensor& Rand();
  virtual Tensor& Uniform(double a = 0.0, double b = 1.0);
  virtual Tensor& UniformReal(double a = 0.0, double b = 1.0);
  virtual Tensor& UniformInt(
      int a = 0, int b = ::std::numeric_limits< int >::max());
  virtual Tensor& Bernoulli(double p = 0.5);
  virtual Tensor& Binomial(int t = 1, double p = 0.5);
  virtual Tensor& NegativeBinomial(int k = 1, double p = 0.5);
  virtual Tensor& Geometric(double p = 0.5);
  virtual Tensor& Poisson(double mean = 1.0);
  virtual Tensor& Exponential(double lambda = 1.0);
  virtual Tensor& Gamma(double alpha = 1.0, double beta = 1.0);
  virtual Tensor& Weibull(double a = 1.0, double b = 1.0);
  virtual Tensor& ExtremeValue(double a = 0.0, double b = 1.0);
  virtual Tensor& Normal(double mean = 0.0, double stddev = 1.0);
  virtual Tensor& LogNormal(double m = 0.0, double s = 1.0);
  virtual Tensor& ChiSquared(double n = 1.0);
  virtual Tensor& Cauchy(double a = 0.0, double b = 1.0);
  virtual Tensor& FisherF(double m = 1.0, double n = 1.0);
  virtual Tensor& StudentT(double n = 1.0);

  // Static random generators are delegated
  static Tensor Rand(const size_storage &size);
  static Tensor Uniform(const size_storage &size, double a = 0.0,
                        double b = 1.0);
  static Tensor UniformReal(const size_storage &size, double a = 0.0,
                            double b = 1.0);
  static Tensor UniformInt(const size_storage &size, int a = 0,
                           int b = ::std::numeric_limits< int >::max());
  static Tensor Bernoulli(const size_storage &size, double p = 0.5);
  static Tensor Binomial(const size_storage &size, int t = 1,
                         double p = 0.5);
  static Tensor NegativeBinomial(const size_storage &size, int k = 1,
                                 double p = 0.5);
  static Tensor Geometric(const size_storage &size, double p = 0.5);
  static Tensor Poisson(const size_storage &size, double mean = 1.0);
  static Tensor Exponential(const size_storage &size, double lambda = 1.0);
  static Tensor Gamma(const size_storage &size, double alpha = 1.0,
                      double beta = 1.0);
  static Tensor Weibull(const size_storage &size, double a = 1.0,
                        double b = 1.0);
  static Tensor ExtremeValue(const size_storage &size, double a = 0.0,
                             double b = 1.0);
  static Tensor Normal(const size_storage &size, double mean = 0.0,
                       double stddev = 1.0);
  static Tensor LogNormal(const size_storage &size, double m = 0.0,
                          double s = 1.0);
  static Tensor ChiSquared(const size_storage &size, double n = 1.0);
  static Tensor Cauchy(const size_storage &size, double a = 0.0,
                       double b = 1.0);
  static Tensor FisherF(const size_storage &size, double m = 1.0,
                        double n = 1.0);
  static Tensor StudentT(const size_storage &size, double n = 1.0);

  // Element-wise mathematical operations that are free of parameters
  virtual const Tensor& Exp() const;
  virtual const Tensor& Exp2() const;
  virtual const Tensor& ExpM1() const;
  virtual const Tensor& Log() const;
  virtual const Tensor& Log10() const;
  virtual const Tensor& Log2() const;
  virtual const Tensor& Log1P() const;
  virtual const Tensor& Sqrt() const;
  virtual const Tensor& Cbrt() const;
  virtual const Tensor& Sin() const;
  virtual const Tensor& Cos() const;
  virtual const Tensor& Tan() const;
  virtual const Tensor& ASin() const;
  virtual const Tensor& ACos() const;
  virtual const Tensor& ATan() const;
  virtual const Tensor& SinH() const;
  virtual const Tensor& CosH() const;
  virtual const Tensor& TanH() const;
  virtual const Tensor& ASinH() const;
  virtual const Tensor& ACosH() const;
  virtual const Tensor& ATanH() const;
  virtual const Tensor& Erf() const;
  virtual const Tensor& ErfC() const;
  virtual const Tensor& TGamma() const;
  virtual const Tensor& LGamma() const;
  virtual const Tensor& Ceil() const;
  virtual const Tensor& Floor() const;
  virtual const Tensor& Trunc() const;
  virtual const Tensor& Round() const;
  virtual const Tensor& NearbyInt() const;
  virtual const Tensor& RInt() const;
  virtual const Tensor& LogB() const;
  virtual const Tensor& FPClassify() const;
  virtual const Tensor& IsFinite() const;
  virtual const Tensor& IsInf() const;
  virtual const Tensor& IsNana() const;
  virtual const Tensor& IsNormal() const;
  virtual const Tensor& Signbit() const;
  virtual const Tensor& Zero() const;

  // Non-const element-wise operations are delegated using const_cast
  virtual Tensor& Exp();
  virtual Tensor& Exp2();
  virtual Tensor& ExpM1();
  virtual Tensor& Log();
  virtual Tensor& Log10();
  virtual Tensor& Log2();
  virtual Tensor& Log1P();
  virtual Tensor& Sqrt();
  virtual Tensor& Cbrt();
  virtual Tensor& Sin();
  virtual Tensor& Cos();
  virtual Tensor& Tan();
  virtual Tensor& ASin();
  virtual Tensor& ACos();
  virtual Tensor& ATan();
  virtual Tensor& SinH();
  virtual Tensor& CosH();
  virtual Tensor& TanH();
  virtual Tensor& ASinH();
  virtual Tensor& ACosH();
  virtual Tensor& ATanH();
  virtual Tensor& Erf();
  virtual Tensor& ErfC();
  virtual Tensor& TGamma();
  virtual Tensor& LGamma();
  virtual Tensor& Ceil();
  virtual Tensor& Floor();
  virtual Tensor& Trunc();
  virtual Tensor& Round();
  virtual Tensor& NearbyInt();
  virtual Tensor& RInt();
  virtual Tensor& LogB();
  virtual Tensor& FPClassify();
  virtual Tensor& IsFinite();
  virtual Tensor& IsInf();
  virtual Tensor& IsNana();
  virtual Tensor& IsNormal();
  virtual Tensor& Signbit();
  virtual Tensor& Zero();

  // static element-wise mathematical operations are deligated
  static Tensor Exp(const Tensor &t);
  static Tensor Exp2(const Tensor &t);
  static Tensor ExpM1(const Tensor &t);
  static Tensor Log(const Tensor &t);
  static Tensor Log10(const Tensor &t);
  static Tensor Log2(const Tensor &t);
  static Tensor Log1P(const Tensor &t);
  static Tensor Sqrt(const Tensor &t);
  static Tensor Cbrt(const Tensor &t);
  static Tensor Sin(const Tensor &t);
  static Tensor Cos(const Tensor &t);
  static Tensor Tan(const Tensor &t);
  static Tensor ASin(const Tensor &t);
  static Tensor ACos(const Tensor &t);
  static Tensor ATan(const Tensor &t);
  static Tensor SinH(const Tensor &t);
  static Tensor CosH(const Tensor &t);
  static Tensor TanH(const Tensor &t);
  static Tensor ASinH(const Tensor &t);
  static Tensor ACosH(const Tensor &t);
  static Tensor ATanH(const Tensor &t);
  static Tensor Erf(const Tensor &t);
  static Tensor ErfC(const Tensor &t);
  static Tensor TGamma(const Tensor &t);
  static Tensor LGamma(const Tensor &t);
  static Tensor Ceil(const Tensor &t);
  static Tensor Floor(const Tensor &t);
  static Tensor Trunc(const Tensor &t);
  static Tensor Round(const Tensor &t);
  static Tensor NearbyInt(const Tensor &t);
  static Tensor RInt(const Tensor &t);
  static Tensor LogB(const Tensor &t);
  static Tensor FPClassify(const Tensor &t);
  static Tensor IsFinite(const Tensor &t);
  static Tensor IsInf(const Tensor &t);
  static Tensor IsNana(const Tensor &t);
  static Tensor IsNormal(const Tensor &t);
  static Tensor Signbit(const Tensor &t);
  static Tensor Zero(const Tensor &t);

  // Element-wise operations with a constant
  virtual const Tensor& Add(const_reference x) const;
  virtual const Tensor& Sub(const_reference x) const;
  virtual const Tensor& Mul(const_reference x) const;
  virtual const Tensor& Div(const_reference x) const;
  virtual const Tensor& FMod(const_reference x) const;
  virtual const Tensor& Remainder(const_reference x) const;
  virtual const Tensor& FMax(const_reference x) const;
  virtual const Tensor& FMin(const_reference x) const;
  virtual const Tensor& FDim(const_reference x) const;
  virtual const Tensor& Pow(const_reference x) const;
  virtual const Tensor& Hypot(const_reference x) const;
  virtual const Tensor& ATan2(const_reference x) const;
  virtual const Tensor& LDExp(const_reference x) const;
  virtual const Tensor& ScalBN(const_reference x) const;
  virtual const Tensor& ScalBLN(const_reference x) const;
  virtual const Tensor& NextAfter(const_reference x) const;
  virtual const Tensor& NextToward(const_reference x) const;
  virtual const Tensor& CopySign(const_reference x) const;
  virtual const Tensor& IsGreater(const_reference x) const;
  virtual const Tensor& IsGreaterEqual(const_reference x) const;
  virtual const Tensor& IsLess(const_reference x) const;
  virtual const Tensor& IsLessEqual(const_reference x) const;
  virtual const Tensor& IsLessGreater(const_reference x) const;
  virtual const Tensor& IsUnordered(const_reference x) const;
  virtual const Tensor& Fill(const_reference x) const;

  // Non-const element-wise operations are delegated using static_cast
  virtual Tensor& Add(const_reference x);
  virtual Tensor& Sub(const_reference x);
  virtual Tensor& Mul(const_reference x);
  virtual Tensor& Div(const_reference x);
  virtual Tensor& FMod(const_reference x);
  virtual Tensor& Remainder(const_reference x);
  virtual Tensor& FMax(const_reference x);
  virtual Tensor& FMin(const_reference x);
  virtual Tensor& FDim(const_reference x);
  virtual Tensor& Pow(const_reference x);
  virtual Tensor& Hypot(const_reference x);
  virtual Tensor& ATan2(const_reference x);
  virtual Tensor& LDExp(const_reference x);
  virtual Tensor& ScalBN(const_reference x);
  virtual Tensor& ScalBLN(const_reference x);
  virtual Tensor& NextAfter(const_reference x);
  virtual Tensor& NextToward(const_reference x);
  virtual Tensor& CopySign(const_reference x);
  virtual Tensor& IsGreater(const_reference x);
  virtual Tensor& IsGreaterEqual(const_reference x);
  virtual Tensor& IsLess(const_reference x);
  virtual Tensor& IsLessEqual(const_reference x);
  virtual Tensor& IsLessGreater(const_reference x);
  virtual Tensor& IsUnordered(const_reference x);
  virtual Tensor& Fill(const_reference x);

  // Static element-wise operations with a constant are delegated
  static Tensor Add(const Tensor &t, const_reference x);
  static Tensor Sub(const Tensor &t, const_reference x);
  static Tensor Mul(const Tensor &t, const_reference x);
  static Tensor Div(const Tensor &t, const_reference x);
  static Tensor FMod(const Tensor &t, const_reference x);
  static Tensor Remainder(const Tensor &t, const_reference x);
  static Tensor FMax(const Tensor &t, const_reference x);
  static Tensor FMin(const Tensor &t, const_reference x);
  static Tensor FDim(const Tensor &t, const_reference x);
  static Tensor Pow(const Tensor &t, const_reference x);
  static Tensor Hypot(const Tensor &t, const_reference x);
  static Tensor ATan2(const Tensor &t, const_reference x);
  static Tensor ScalBN(const Tensor &t, const_reference x);
  static Tensor ScalBLN(const Tensor &t, const_reference x);
  static Tensor NextAfter(const Tensor &t, const_reference x);
  static Tensor NextToward(const Tensor &t, const_reference x);
  static Tensor CopySign(const Tensor &t, const_reference x);
  static Tensor IsGreater(const Tensor &t, const_reference x);
  static Tensor IsGreaterEqual(const Tensor &t, const_reference x);
  static Tensor IsLess(const Tensor &t, const_reference x);
  static Tensor IsLessEqual(const Tensor &t, const_reference x);
  static Tensor IsLessGreater(const Tensor &t, const_reference x);
  static Tensor IsUnordered(const Tensor &t, const_reference x);
  static Tensor Fill(const Tensor &t, const_reference x);

  // Templated element-wise operations with another tensor
  template < typename Other_S >
  const Tensor& Add(const Tensor< Other_S > &x) const;
  template < typename Other_S >
  const Tensor& Sub(const Tensor< Other_S > &x) const;
  template < typename Other_S >
  const Tensor& Mul(const Tensor< Other_S > &x) const;
  template < typename Other_S >
  const Tensor& Div(const Tensor< Other_S > &x) const;
  template < typename Other_S >
  const Tensor& FMod(const Tensor< Other_S > &x) const;
  template < typename Other_S >
  const Tensor& Remainder(const Tensor< Other_S > &x) const;
  template < typename Other_S >
  const Tensor& FMax(const Tensor< Other_S > &x) const;
  template < typename Other_S >
  const Tensor& FMin(const Tensor< Other_S > &x) const;
  template < typename Other_S >
  const Tensor& FDim(const Tensor< Other_S > &x) const;
  template < typename Other_S >
  const Tensor& Pow(const Tensor< Other_S > &x) const;
  template < typename Other_S >
  const Tensor& Hypot(const Tensor< Other_S > &x) const;
  template < typename Other_S >
  const Tensor& ATan2(const Tensor< Other_S > &x) const;
  template < typename Other_S >
  const Tensor& ScalBN(const Tensor< Other_S > &x) const;
  template < typename Other_S >
  const Tensor& ScaleBLN(const Tensor< Other_S > &x) const;
  template < typename Other_S >
  const Tensor& NextAfter(const Tensor< Other_S > &x) const;
  template < typename Other_S >
  const Tensor& NextToward(const Tensor< Other_S > &x) const;
  template < typename Other_S >
  const Tensor& CopySign(const Tensor< Other_S > &x) const;
  template < typename Other_S >
  const Tensor& IsGreater(const Tensor< Other_S > &x) const;
  template < typename Other_S >
  const Tensor& IsGreaterEqual(const Tensor< Other_S > &x) const;
  template < typename Other_S >
  const Tensor& IsLess(const Tensor< Other_S > &x) const;
  template < typename Other_S >
  const Tensor& IsLessEqual(const Tensor< Other_S > &x) const;
  template < typename Other_S >
  const Tensor& IsLessGreater(const Tensor< Other_S > &x) const;
  template < typename Other_S >
  const Tensor& IsUnordered(const Tensor< Other_S > &x) const;

  // Element-wise operations with another tensor
  virtual const Tensor& Add(const Tensor &x) const;
  virtual const Tensor& Sub(const Tensor &x) const;
  virtual const Tensor& Mul(const Tensor &x) const;
  virtual const Tensor& Div(const Tensor &x) const;
  virtual const Tensor& FMod(const Tensor &x) const;
  virtual const Tensor& Remainder(const Tensor &x) const;
  virtual const Tensor& FMax(const Tensor &x) const;
  virtual const Tensor& FMin(const Tensor &x) const;
  virtual const Tensor& FDim(const Tensor &x) const;
  virtual const Tensor& Pow(const Tensor &x) const;
  virtual const Tensor& Hypot(const Tensor &x) const;
  virtual const Tensor& ATan2(const Tensor &x) const;
  virtual const Tensor& ScalBN(const Tensor &x) const;
  virtual const Tensor& ScaleBLN(const Tensor &x) const;
  virtual const Tensor& NextAfter(const Tensor &x) const;
  virtual const Tensor& NextToward(const Tensor &x) const;
  virtual const Tensor& CopySign(const Tensor &x) const;
  virtual const Tensor& IsGreater(const Tensor &x) const;
  virtual const Tensor& IsGreaterEqual(const Tensor &x) const;
  virtual const Tensor& IsLess(const Tensor &x) const;
  virtual const Tensor& IsLessEqual(const Tensor &x) const;
  virtual const Tensor& IsLessGreater(const Tensor &x) const;
  virtual const Tensor& IsUnordered(const Tensor &x) const;

  // Non-const templated element-wise operations are delegated using const_cast
  template < typename Other_S >
  Tensor& Add(const Tensor< Other_S > &x);
  template < typename Other_S >
  Tensor& Sub(const Tensor< Other_S > &x);
  template < typename Other_S >
  Tensor& Mul(const Tensor< Other_S > &x);
  template < typename Other_S >
  Tensor& Div(const Tensor< Other_S > &x);
  template < typename Other_S >
  Tensor& FMod(const Tensor< Other_S > &x);
  template < typename Other_S >
  Tensor& Remainder(const Tensor< Other_S > &x);
  template < typename Other_S >
  Tensor& FMax(const Tensor< Other_S > &x);
  template < typename Other_S >
  Tensor& FMin(const Tensor< Other_S > &x);
  template < typename Other_S >
  Tensor& FDim(const Tensor< Other_S > &x);
  template < typename Other_S >
  Tensor& Pow(const Tensor< Other_S > &x);
  template < typename Other_S >
  Tensor& Hypot(const Tensor< Other_S > &x);
  template < typename Other_S >
  Tensor& ATan2(const Tensor< Other_S > &x);
  template < typename Other_S >
  Tensor& ScalBN(const Tensor< Other_S > &x);
  template < typename Other_S >
  Tensor& ScaleBLN(const Tensor< Other_S > &x);
  template < typename Other_S >
  Tensor& NextAfter(const Tensor< Other_S > &x);
  template < typename Other_S >
  Tensor& NextToward(const Tensor< Other_S > &x);
  template < typename Other_S >
  Tensor& CopySign(const Tensor< Other_S > &x);
  template < typename Other_S >
  Tensor& IsGreater(const Tensor< Other_S > &x);
  template < typename Other_S >
  Tensor& IsGreaterEqual(const Tensor< Other_S > &x);
  template < typename Other_S >
  Tensor& IsLess(const Tensor< Other_S > &x);
  template < typename Other_S >
  Tensor& IsLessEqual(const Tensor< Other_S > &x);
  template < typename Other_S >
  Tensor& IsLessGreater(const Tensor< Other_S > &x);
  template < typename Other_S >
  Tensor& IsUnordered(const Tensor< Other_S > &x);

  // Non-const element-wise operations are delegated using const_cast
  virtual Tensor& Add(const Tensor &x);
  virtual Tensor& Sub(const Tensor &x);
  virtual Tensor& Mul(const Tensor &x);
  virtual Tensor& Div(const Tensor &x);
  virtual Tensor& FMod(const Tensor &x);
  virtual Tensor& Remainder(const Tensor &x);
  virtual Tensor& FMax(const Tensor &x);
  virtual Tensor& FMin(const Tensor &x);
  virtual Tensor& FDim(const Tensor &x);
  virtual Tensor& Pow(const Tensor &x);
  virtual Tensor& Hypot(const Tensor &x);
  virtual Tensor& ATan2(const Tensor &x);
  virtual Tensor& ScalBN(const Tensor &x);
  virtual Tensor& ScaleBLN(const Tensor &x);
  virtual Tensor& NextAfter(const Tensor &x);
  virtual Tensor& NextToward(const Tensor &x);
  virtual Tensor& CopySign(const Tensor &x);
  virtual Tensor& IsGreater(const Tensor &x);
  virtual Tensor& IsGreaterEqual(const Tensor &x);
  virtual Tensor& IsLess(const Tensor &x);
  virtual Tensor& IsLessEqual(const Tensor &x);
  virtual Tensor& IsLessGreater(const Tensor &x);
  virtual Tensor& IsUnordered(const Tensor &x);

  // Static templated element-wise operations with another tensor are delegated
  template < typename Other_S >
  static Tensor Add(const Tensor &t, const Tensor< Other_S > &x);
  template < typename Other_S >
  static Tensor Sub(const Tensor &t, const Tensor< Other_S > &x);
  template < typename Other_S >
  static Tensor Mul(const Tensor &t, const Tensor< Other_S > &x);
  template < typename Other_S >
  static Tensor Div(const Tensor &t, const Tensor< Other_S > &x);
  template < typename Other_S >
  static Tensor FMod(const Tensor &t, const Tensor< Other_S > &x);
  template < typename Other_S >
  static Tensor Remainder(const Tensor &t, const Tensor< Other_S > &x);
  template < typename Other_S >
  static Tensor FMax(const Tensor &t, const Tensor< Other_S > &x);
  template < typename Other_S >
  static Tensor FMin(const Tensor &t, const Tensor< Other_S > &x);
  template < typename Other_S >
  static Tensor FDim(const Tensor &t, const Tensor< Other_S > &x);
  template < typename Other_S >
  static Tensor Pow(const Tensor &t, const Tensor< Other_S > &x);
  template < typename Other_S >
  static Tensor Hypot(const Tensor &t, const Tensor< Other_S > &x);
  template < typename Other_S >
  static Tensor ATan2(const Tensor &t, const Tensor< Other_S > &x);
  template < typename Other_S >
  static Tensor ScalBN(const Tensor &t, const Tensor< Other_S > &x);
  template < typename Other_S >
  static Tensor ScaleBLN(const Tensor &t, const Tensor< Other_S > &x);
  template < typename Other_S >
  static Tensor NextAfter(const Tensor &t, const Tensor< Other_S > &x);
  template < typename Other_S >
  static Tensor NextToward(const Tensor &t, const Tensor< Other_S > &x);
  template < typename Other_S >
  static Tensor CopySign(const Tensor &t, const Tensor< Other_S > &x);
  template < typename Other_S >
  static Tensor IsGreater(const Tensor &t, const Tensor< Other_S > &x);
  template < typename Other_S >
  static Tensor IsGreaterEqual(const Tensor &t, const Tensor< Other_S > &x);
  template < typename Other_S >
  static Tensor IsLess(const Tensor &t, const Tensor< Other_S > &x);
  template < typename Other_S >
  static Tensor IsLessEqual(const Tensor &t, const Tensor< Other_S > &x);
  template < typename Other_S >
  static Tensor IsLessGreater(const Tensor &t, const Tensor< Other_S > &x);
  template < typename Other_S >
  static Tensor IsUnordered(const Tensor &t, const Tensor< Other_S > &x);

  // Static element-wise operations with another tensor are delegated
  static Tensor Add(const Tensor &t, const Tensor &x);
  static Tensor Sub(const Tensor &t, const Tensor &x);
  static Tensor Mul(const Tensor &t, const Tensor &x);
  static Tensor Div(const Tensor &t, const Tensor &x);
  static Tensor FMod(const Tensor &t, const Tensor &x);
  static Tensor Remainder(const Tensor &t, const Tensor &x);
  static Tensor FMax(const Tensor &t, const Tensor &x);
  static Tensor FMin(const Tensor &t, const Tensor &x);
  static Tensor FDim(const Tensor &t, const Tensor &x);
  static Tensor Pow(const Tensor &t, const Tensor &x);
  static Tensor Hypot(const Tensor &t, const Tensor &x);
  static Tensor ATan2(const Tensor &t, const Tensor &x);
  static Tensor ScalBN(const Tensor &t, const Tensor &x);
  static Tensor ScaleBLN(const Tensor &t, const Tensor &x);
  static Tensor NextAfter(const Tensor &t, const Tensor &x);
  static Tensor NextToward(const Tensor &t, const Tensor &x);
  static Tensor CopySign(const Tensor &t, const Tensor &x);
  static Tensor IsGreater(const Tensor &t, const Tensor &x);
  static Tensor IsGreaterEqual(const Tensor &t, const Tensor &x);
  static Tensor IsLess(const Tensor &t, const Tensor &x);
  static Tensor IsLessEqual(const Tensor &t, const Tensor &x);
  static Tensor IsLessGreater(const Tensor &t, const Tensor &x);
  static Tensor IsUnordered(const Tensor &t, const Tensor &x);

  // Fused operations with values and tensors
  const Tensor& FMA(const_reference y, const_reference z) const;
  const Tensor& FMA(const Tensor& y, const_reference z) const;
  const Tensor& FMA(const Tensor& y, const Tensor& z) const;

  // Non-const fused operations are delegated using const_cast
  Tensor& FMA(const_reference y, const_reference z);
  Tensor& FMA(const Tensor &y, const_reference z);
  Tensor& FMA(const Tensor &y, const Tensor& z);

  // Static fused operations are delegated
  static Tensor FMA(const Tensor &x, const_reference y, const_reference z);
  static Tensor FMA(const Tensor &x, const Tensor &y, const_reference z);
  static Tensor FMA(const Tensor &x, const Tensor &y, const Tensor &z);

  // Constructor functions can only be static
  static Tensor Cat(const Tensor& s, const Tensor& t, dim_type dim);
  static Tensor Diag(const Tensor& s);
  static Tensor Eye(size_type n);
  static Tensor Eye(size_type m, size_type n);
  static Tensor Eye(const size_storage &size);
  static Tensor LinSpace(const_reference x, const_reference y, size_type n);
  static Tensor LogSpace(const_reference x, const_reference y, size_type n);
  static Tensor Ones(size_type n);
  static Tensor Ones(size_type m, size_type n);
  static Tensor Ones(size_type n0, size_type n1, size_type n2);
  static Tensor Ones(size_type n0, size_type n1, size_type n2, size_type n3);
  static Tensor Ones(const size_storage &size);
  static Tensor Range(const_reference x, const_reference y, reference step);
  static Tensor Reshape(const Tensor& tensor, const size_storage size);
  static Tensor TriL(const Tensor &s);
  static Tensor TriU(const Tensor &s);
  static Tensor Zeros(size_type n);
  static Tensor Zeros(size_type m, size_type n);
  static Tensor Zeros(size_type n0, size_type n1, size_type n2);
  static Tensor Zeros(size_type n0, size_type n1, size_type n2, size_type n3);
  static Tensor Zeros(const size_storage &size);

 private:
  size_storage size_;
  stride_storage stride_;
  storage_pointer storage_;
  size_type offset_;
};

}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_HPP
