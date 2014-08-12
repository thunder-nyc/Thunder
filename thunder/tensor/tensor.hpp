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
  virtual operator value_type() const;

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
  friend Tensor operator+(const_reference value, const Tensor &t);
  friend Tensor operator-(const_reference value, const Tensor &t);

  // Arithmetic operators are delegated
  virtual Tensor operator+(const Tensor &other) const;
  virtual Tensor operator-(const Tensor &other) const;

  // Comparison operators with value are delegated
  virtual Tensor operator==(const_reference value) const;
  virtual Tensor operator>(const_reference value) const;
  virtual Tensor operator<(const_reference value) const;
  virtual Tensor operator<=(const_reference value) const;
  virtual Tensor operator>=(const_reference value) const;
  friend Tensor operator==(const_reference value, const Tensor &t);
  friend Tensor operator>(const_reference value, const Tensor &t);
  friend Tensor operator<(const_reference value, const Tensor &t);
  friend Tensor operator>=(const_reference value, const Tensor &t);
  friend Tensor operator<=(const_reference value, const Tensor &t);

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
  template < typename T >
  bool isSameSizeAs(const T &other) const;

  // Static non-virtual templated queriers are delegated
  template < typename T >
  static bool isSameSizeAs(const Tensor& t, const T &other);

  // Property queries
  virtual dim_type dimension() const;
  virtual size_storage size() const;
  virtual size_type size(dim_type dim) const;
  virtual size_type count() const;
  virtual stride_storage stride() const;
  virtual difference_type stride(dim_type dim) const;
  virtual storage_pointer storage() const;
  virtual size_type offset() const;
  virtual pointer data() const;
  virtual bool isContiguous() const;
  virtual bool isSameSizeAs(const Tensor &other) const;

  // Static property queries are delegated
  static dim_type dimension(const Tensor &t);
  static size_storage size(const Tensor &t);
  static size_type size(const Tensor &t, dim_type dim);
  static size_type count(const Tensor &t);
  static stride_storage stride(const Tensor &t);
  static difference_type stride(const Tensor &t, dim_type dim);
  static storage_pointer getStorage(const Tensor &t);
  static size_type offset(const Tensor &t);
  static pointer data(const Tensor &t);
  static bool isContiguous(const Tensor &t);
  static bool isSameSizeAs(const Tensor &t, const Tensor &other);

  // Non-virtual templated modifiers
  template < typename T >
  Tensor& copy(const T &other);
  template < typename T >
  Tensor& resizeAs(const T &other);
  virtual Tensor& copy(const Tensor &other);

  // Static non-virtual templated modifiers are delegated
  template < typename T >
  static Tensor& copy(Tensor *t, const T &other);
  static Tensor& copy(Tensor *t, const Tensor &other);
  template < typename T >
  static Tensor& resizeAs(Tensor *t, const T &other);
  static Tensor& resizeAs(Tensor *t, const Tensor &other);

  // Normal modifiers
  virtual Tensor& set(const Tensor &other);
  virtual Tensor& set(const storage_pointer &storage, size_type offset = 0);
  virtual Tensor& set(const size_storage &size, const storage_pointer &storage,
                      size_type offset = 0);
  virtual Tensor& set(const size_storage &size, const stride_storage &stride,
                      const storage_pointer &storage, size_type offset = 0);
  virtual Tensor& resize(const size_storage &size);
  virtual Tensor& resize(const size_storage &size,
                         const stride_storage &stride);
  virtual Tensor& resizeAs(const Tensor &other);

  // Static modifiers are delegated
  static Tensor& set(Tensor *t, const Tensor &other);
  static Tensor& set(Tensor *t, const storage_pointer &storage,
                     size_type offset = 0);
  static Tensor& set(Tensor *t, const size_storage &size,
                     const storage_pointer &storage, size_type offset = 0);
  static Tensor& set(Tensor *t, const size_storage &size,
                     const stride_storage &stride,
                     const storage_pointer &storage, size_type offset = 0);
  static Tensor& resize(Tensor *t, const size_storage &size);
  static Tensor& resize(Tensor *t, const size_storage &size,
                        const stride_storage &stride);

  // Templated subtensor extractors
  template < typename T >
  Tensor viewAs(const T &other) const;
  virtual Tensor viewAs(const Tensor &other) const;

  // Static templated subtensor extractors are delegated
  template < typename T >
  static Tensor viewAs(const Tensor& t, const T &other);
  static Tensor viewAs(const Tensor &t, const Tensor &other);

  // Extract subtensors or transformations -- no need for non-const overload
  virtual Tensor narrow(dim_type dim, size_type pos, size_type size) const;
  virtual Tensor select(dim_type dim, size_type pos) const;
  virtual Tensor view(size_type size0) const;
  virtual Tensor view(size_type size0, size_type size1) const;
  virtual Tensor view(size_type size0, size_type size1, size_type size2) const;
  virtual Tensor view(size_type size0, size_type size1, size_type size2,
                      size_type size3) const;
  virtual Tensor view(const size_storage &size) const;
  virtual Tensor transpose() const;
  virtual Tensor transpose(dim_type dim0, dim_type dim1) const;
  virtual Tensor unfold(dim_type dim, size_type size, size_type step) const;
  virtual Tensor clone() const;
  virtual Tensor cat(const Tensor& t, dim_type dim) const;
  virtual Tensor diag() const;
  virtual Tensor reshape(size_type s0) const;
  virtual Tensor reshape(size_type s0, size_type s1) const;
  virtual Tensor reshape(size_type s0, size_type s1, size_type s2) const;
  virtual Tensor reshape(size_type s0, size_type s1, size_type s2,
                         size_type s3) const;
  virtual Tensor reshape(const size_storage size) const;
  virtual Tensor triL() const;
  virtual Tensor triU() const;
  virtual Tensor contiguous();

  // Static subtensor or transformation extractors are delegated
  static Tensor narrow(const Tensor &t, dim_type dim, size_type pos,
                       size_type size);
  static Tensor select(const Tensor &t, dim_type dim, size_type pos);
  static Tensor view(const Tensor &t, size_type size0);
  static Tensor view(const Tensor &t, size_type size0, size_type size1);
  static Tensor view(const Tensor &t, size_type size0, size_type size1,
                     size_type size2);
  static Tensor view(const Tensor &t, size_type size0, size_type size1,
                     size_type size2, size_type size3);
  static Tensor view(const Tensor &t, const size_storage &size);
  static Tensor transpose(const Tensor &t);
  static Tensor transpose(const Tensor &t, dim_type dim0, dim_type dim1);
  static Tensor unfold(const Tensor &t, dim_type dim, size_type size,
                       size_type step);
  static Tensor clone(const Tensor& t);
  static Tensor cat(const Tensor &s, const Tensor& t, dim_type dim);
  static Tensor diag(const Tensor &s);
  static Tensor reshape(const Tensor &s, size_type s0);
  static Tensor reshape(const Tensor &s, size_type s0, size_type s1);
  static Tensor reshape(const Tensor &s, size_type s0, size_type s1,
                        size_type s2);
  static Tensor reshape(const Tensor &s, size_type s0, size_type s1,
                        size_type s2, size_type s3);
  static Tensor reshape(const Tensor &s, const size_storage size);
  static Tensor triL(const Tensor &s);
  static Tensor triU(const Tensor &s);
  static Tensor contiguous(const Tensor &t);

  // lambda applications
  virtual const Tensor& apply(
      const ::std::function< value_type(value_type) > &lambda) const;
  virtual const Tensor& apply(
      const ::std::function< value_type(const value_type&) > &lambda) const;
  virtual const Tensor& apply(
      const ::std::function< void(value_type&) > &lambda) const;
  virtual const Tensor& apply(
      const ::std::function< void(value_type*) > &lambda) const;

  // Non-const lambda applications are delegated using const_cast
  virtual Tensor& apply(
      const ::std::function< value_type(value_type) > &lambda);
  virtual Tensor& apply(
      const ::std::function< value_type(const value_type&) > &lambda);
  virtual Tensor& apply(const ::std::function< void(value_type&) > &lambda);
  virtual Tensor& apply(const ::std::function< void(value_type*) > &lambda);

  // Static lambda applications are delegated
  static Tensor apply(const Tensor& t,
                      const ::std::function< value_type(value_type) > &lambda);
  static Tensor apply(
      const Tensor& t,
      const ::std::function< value_type(const value_type&) > &lambda);
  static Tensor apply(const Tensor& t,
                      const ::std::function< void(value_type&) > &lambda);
  static Tensor apply(const Tensor& t,
                      const ::std::function< void(value_type*) > &lambda);

  // Reduction operations
  virtual value_type max() const;
  virtual value_type min() const;
  virtual value_type sum() const;
  virtual value_type mean() const;
  virtual value_type var() const;
  virtual value_type std() const;

  // Static reduction operations are deligated
  static value_type max(const Tensor &t);
  static value_type min(const Tensor &t);
  static value_type sum(const Tensor &t);
  static value_type mean(const Tensor &t);
  static value_type var(const Tensor &t);
  static value_type std(const Tensor &t);

  // All random generators are templated on RF and RI.
  template < typename RF = double >
  const Tensor& uniform(RF a = 0.0, RF b = 1.0) const;
  template < typename RF = double >
  const Tensor& uniformReal(RF a = 0.0, RF b = 1.0) const;
  template < typename RI = int >
  const Tensor& uniformInt(RI a = 0,
                           RI b = ::std::numeric_limits< RI >::max()) const;
  template < typename RF = double >
  const Tensor& bernoulli(RF p = 0.5) const;
  template < typename RI = int, typename RF = double >
  const Tensor& ninomial(RI t = 1, RF p = 0.5) const;
  template < typename RI = int, typename RF = double >
  const Tensor& negativeBinomial(RI k = 1, RF p = 0.5) const;
  template < typename RF = double >
  const Tensor& geometric(RF p = 0.5) const;
  template < typename RF = double >
  const Tensor& poisson(RF mean = 1.0) const;
  template < typename RF = double >
  const Tensor& exponential(RF lambda = 1.0) const;
  template < typename RF = double >
  const Tensor& gamma(RF alpha = 1.0, RF beta = 1.0) const;
  template < typename RF = double >
  const Tensor& weibull(RF a = 1.0, RF b = 1.0) const;
  template < typename RF = double >
  const Tensor& extremeValue(RF a = 0.0, RF b = 1.0) const;
  template < typename RF = double >
  const Tensor& normal(RF mean = 0.0, RF stddev = 1.0) const;
  template < typename RF = double >
  const Tensor& logNormal(RF m = 0.0, RF s = 1.0) const;
  template < typename RF = double >
  const Tensor& chiSquared(RF n = 1.0) const;
  template < typename RF = double >
  const Tensor& cauchy(RF a = 0.0, RF b = 1.0) const;
  template < typename RF = double >
  const Tensor& fisherF(RF m = 1.0, RF n = 1.0) const;
  template < typename RF = double >
  const Tensor& studentT(RF n = 1.0) const;

  // Non-const random generators are delegated using const_cast.
  virtual Tensor& rand();
  template < typename RF = double >
  Tensor& uniform(RF a = 0.0, RF b = 1.0);
  template < typename RF = double >
  Tensor& uniformReal(RF a = 0.0, RF b = 1.0);
  template < typename RI = int >
  Tensor& uniformInt(RI a = 0,
                     RI b = ::std::numeric_limits< RI >::max());
  template < typename RF = double >
  Tensor& bernoulli(RF p = 0.5);
  template < typename RI = int, typename RF = double >
  Tensor& ninomial(RI t = 1, RF p = 0.5);
  template < typename RI = int, typename RF = double >
  Tensor& negativeBinomial(RI k = 1, RF p = 0.5);
  template < typename RF = double >
  Tensor& geometric(RF p = 0.5);
  template < typename RF = double >
  Tensor& poisson(RF mean = 1.0);
  template < typename RF = double >
  Tensor& exponential(RF lambda = 1.0);
  template < typename RF = double >
  Tensor& gamma(RF alpha = 1.0, RF beta = 1.0);
  template < typename RF = double >
  Tensor& weibull(RF a = 1.0, RF b = 1.0);
  template < typename RF = double >
  Tensor& extremeValue(RF a = 0.0, RF b = 1.0);
  template < typename RF = double >
  Tensor& normal(RF mean = 0.0, RF stddev = 1.0);
  template < typename RF = double >
  Tensor& logNormal(RF m = 0.0, RF s = 1.0);
  template < typename RF = double >
  Tensor& chiSquared(RF n = 1.0);
  template < typename RF = double >
  Tensor& cauchy(RF a = 0.0, RF b = 1.0);
  template < typename RF = double >
  Tensor& fisherF(RF m = 1.0, RF n = 1.0);
  template < typename RF = double >
  Tensor& studentT(RF n = 1.0);

  // Static random generators are delegated
  static Tensor rand(const size_storage &size);
  template < typename RF = double >
  static Tensor uniform(const size_storage &size, RF a = 0.0, RF b = 1.0);
  template < typename RF = double >
  static Tensor uniformReal(const size_storage &size, RF a = 0.0, RF b = 1.0);
  template < typename RI = int >
  static Tensor uniformInt(const size_storage &size, RI a = 0,
                           RI b = ::std::numeric_limits< RI >::max());
  template < typename RF = double >
  static Tensor bernoulli(const size_storage &size, RF p = 0.5);
  template < typename RI = int, typename RF = double >
  static Tensor ninomial(const size_storage &size, RI t = 1, RF p = 0.5);
  template < typename RI = int, typename RF = double >
  static Tensor negativeBinomial(const size_storage &size, RI k = 1,
                                 RF p = 0.5);
  template < typename RF = double >
  static Tensor geometric(const size_storage &size, RF p = 0.5);
  template < typename RF = double >
  static Tensor poisson(const size_storage &size, RF mean = 1.0);
  template < typename RF = double >
  static Tensor exponential(const size_storage &size, RF lambda = 1.0);
  template < typename RF = double >
  static Tensor gamma(const size_storage &size, RF alpha = 1.0, RF beta = 1.0);
  template < typename RF = double >
  static Tensor weibull(const size_storage &size, RF a = 1.0, RF b = 1.0);
  template < typename RF = double >
  static Tensor extremeValue(const size_storage &size, RF a = 0.0, RF b = 1.0);
  template < typename RF = double >
  static Tensor normal(const size_storage &size, RF mean = 0.0,
                       RF stddev = 1.0);
  template < typename RF = double >
  static Tensor logNormal(const size_storage &size, RF m = 0.0, RF s = 1.0);
  template < typename RF = double >
  static Tensor chiSquared(const size_storage &size, RF n = 1.0);
  template < typename RF = double >
  static Tensor cauchy(const size_storage &size, RF a = 0.0, RF b = 1.0);
  template < typename RF = double >
  static Tensor fisherF(const size_storage &size, RF m = 1.0, RF n = 1.0);
  template < typename RF = double >
  static Tensor studentT(const size_storage &size, RF n = 1.0);

  // Element-wise mathematical operations that are free of parameters
  virtual const Tensor& exp() const;
  virtual const Tensor& exp2() const;
  virtual const Tensor& expm1() const;
  virtual const Tensor& log() const;
  virtual const Tensor& log10() const;
  virtual const Tensor& log2() const;
  virtual const Tensor& log1p() const;
  virtual const Tensor& sqrt() const;
  virtual const Tensor& cbrt() const;
  virtual const Tensor& sin() const;
  virtual const Tensor& cos() const;
  virtual const Tensor& tan() const;
  virtual const Tensor& asin() const;
  virtual const Tensor& acos() const;
  virtual const Tensor& atan() const;
  virtual const Tensor& sinh() const;
  virtual const Tensor& cosh() const;
  virtual const Tensor& tanh() const;
  virtual const Tensor& asinh() const;
  virtual const Tensor& acosh() const;
  virtual const Tensor& atanh() const;
  virtual const Tensor& erf() const;
  virtual const Tensor& erfc() const;
  virtual const Tensor& tgamma() const;
  virtual const Tensor& lgamma() const;
  virtual const Tensor& ceil() const;
  virtual const Tensor& floor() const;
  virtual const Tensor& trunc() const;
  virtual const Tensor& round() const;
  virtual const Tensor& nearbyint() const;
  virtual const Tensor& rint() const;
  virtual const Tensor& logb() const;
  virtual const Tensor& fpclassify() const;
  virtual const Tensor& isfinite() const;
  virtual const Tensor& isinf() const;
  virtual const Tensor& isnan() const;
  virtual const Tensor& isnormal() const;
  virtual const Tensor& signbit() const;
  virtual const Tensor& zero() const;

  // Non-const element-wise operations are delegated using const_cast
  virtual Tensor& exp();
  virtual Tensor& exp2();
  virtual Tensor& expm1();
  virtual Tensor& log();
  virtual Tensor& log10();
  virtual Tensor& log2();
  virtual Tensor& log1p();
  virtual Tensor& sqrt();
  virtual Tensor& cbrt();
  virtual Tensor& sin();
  virtual Tensor& cos();
  virtual Tensor& tan();
  virtual Tensor& asin();
  virtual Tensor& acos();
  virtual Tensor& atan();
  virtual Tensor& sinh();
  virtual Tensor& cosh();
  virtual Tensor& tanh();
  virtual Tensor& asinh();
  virtual Tensor& acosh();
  virtual Tensor& atanh();
  virtual Tensor& erf();
  virtual Tensor& erfc();
  virtual Tensor& tgamma();
  virtual Tensor& lgamma();
  virtual Tensor& ceil();
  virtual Tensor& floor();
  virtual Tensor& trunc();
  virtual Tensor& round();
  virtual Tensor& nearbyint();
  virtual Tensor& rint();
  virtual Tensor& logB();
  virtual Tensor& fpclassify();
  virtual Tensor& isfinite();
  virtual Tensor& isinf();
  virtual Tensor& isnan();
  virtual Tensor& isnormal();
  virtual Tensor& signbit();
  virtual Tensor& zero();

  // static element-wise mathematical operations are deligated
  static Tensor exp(const Tensor &t);
  static Tensor exp2(const Tensor &t);
  static Tensor expm1(const Tensor &t);
  static Tensor log(const Tensor &t);
  static Tensor log10(const Tensor &t);
  static Tensor log2(const Tensor &t);
  static Tensor log1p(const Tensor &t);
  static Tensor sqrt(const Tensor &t);
  static Tensor cbrt(const Tensor &t);
  static Tensor sin(const Tensor &t);
  static Tensor cos(const Tensor &t);
  static Tensor tan(const Tensor &t);
  static Tensor asin(const Tensor &t);
  static Tensor acos(const Tensor &t);
  static Tensor atan(const Tensor &t);
  static Tensor sinh(const Tensor &t);
  static Tensor cosh(const Tensor &t);
  static Tensor tanh(const Tensor &t);
  static Tensor asinh(const Tensor &t);
  static Tensor acosh(const Tensor &t);
  static Tensor atanh(const Tensor &t);
  static Tensor erf(const Tensor &t);
  static Tensor erfc(const Tensor &t);
  static Tensor tgamma(const Tensor &t);
  static Tensor lgamma(const Tensor &t);
  static Tensor ceil(const Tensor &t);
  static Tensor floor(const Tensor &t);
  static Tensor trunc(const Tensor &t);
  static Tensor round(const Tensor &t);
  static Tensor nearbyint(const Tensor &t);
  static Tensor rint(const Tensor &t);
  static Tensor logb(const Tensor &t);
  static Tensor fpclassify(const Tensor &t);
  static Tensor isfinite(const Tensor &t);
  static Tensor isinf(const Tensor &t);
  static Tensor isnan(const Tensor &t);
  static Tensor isnormal(const Tensor &t);
  static Tensor signbit(const Tensor &t);
  static Tensor zero(const Tensor &t);

  // Element-wise operations with a constant
  virtual const Tensor& add(const_reference x) const;
  virtual const Tensor& sub(const_reference x) const;
  virtual const Tensor& mul(const_reference x) const;
  virtual const Tensor& div(const_reference x) const;
  virtual const Tensor& fmod(const_reference x) const;
  virtual const Tensor& remainder(const_reference x) const;
  virtual const Tensor& fmax(const_reference x) const;
  virtual const Tensor& fmin(const_reference x) const;
  virtual const Tensor& fdim(const_reference x) const;
  virtual const Tensor& pow(const_reference x) const;
  virtual const Tensor& hypot(const_reference x) const;
  virtual const Tensor& atan2(const_reference x) const;
  virtual const Tensor& ldexp(const_reference x) const;
  virtual const Tensor& Scalbn(const_reference x) const;
  virtual const Tensor& Scalbln(const_reference x) const;
  virtual const Tensor& nextafter(const_reference x) const;
  virtual const Tensor& nexttoward(const_reference x) const;
  virtual const Tensor& copysign(const_reference x) const;
  virtual const Tensor& isgreater(const_reference x) const;
  virtual const Tensor& isgreaterequal(const_reference x) const;
  virtual const Tensor& isless(const_reference x) const;
  virtual const Tensor& islessequal(const_reference x) const;
  virtual const Tensor& islessgreater(const_reference x) const;
  virtual const Tensor& isunordered(const_reference x) const;
  virtual const Tensor& fill(const_reference x) const;

  // Non-const element-wise operations are delegated using static_cast
  virtual Tensor& add(const_reference x);
  virtual Tensor& sub(const_reference x);
  virtual Tensor& mul(const_reference x);
  virtual Tensor& div(const_reference x);
  virtual Tensor& fmod(const_reference x);
  virtual Tensor& remainder(const_reference x);
  virtual Tensor& fmax(const_reference x);
  virtual Tensor& fmin(const_reference x);
  virtual Tensor& fdim(const_reference x);
  virtual Tensor& pow(const_reference x);
  virtual Tensor& hypot(const_reference x);
  virtual Tensor& atan2(const_reference x);
  virtual Tensor& ldexp(const_reference x);
  virtual Tensor& scalbn(const_reference x);
  virtual Tensor& scalbln(const_reference x);
  virtual Tensor& nextafter(const_reference x);
  virtual Tensor& nexttoward(const_reference x);
  virtual Tensor& copysign(const_reference x);
  virtual Tensor& isgreater(const_reference x);
  virtual Tensor& isgreaterequal(const_reference x);
  virtual Tensor& isless(const_reference x);
  virtual Tensor& islessequal(const_reference x);
  virtual Tensor& islessgreater(const_reference x);
  virtual Tensor& isunordered(const_reference x);
  virtual Tensor& fill(const_reference x);

  // Static element-wise operations with a constant are delegated
  static Tensor add(const Tensor &t, const_reference x);
  static Tensor sub(const Tensor &t, const_reference x);
  static Tensor mul(const Tensor &t, const_reference x);
  static Tensor div(const Tensor &t, const_reference x);
  static Tensor fmod(const Tensor &t, const_reference x);
  static Tensor remainder(const Tensor &t, const_reference x);
  static Tensor fmax(const Tensor &t, const_reference x);
  static Tensor fmin(const Tensor &t, const_reference x);
  static Tensor fdim(const Tensor &t, const_reference x);
  static Tensor pow(const Tensor &t, const_reference x);
  static Tensor hypot(const Tensor &t, const_reference x);
  static Tensor atan2(const Tensor &t, const_reference x);
  static Tensor scalbn(const Tensor &t, const_reference x);
  static Tensor scalbln(const Tensor &t, const_reference x);
  static Tensor nextafter(const Tensor &t, const_reference x);
  static Tensor nexttoward(const Tensor &t, const_reference x);
  static Tensor copysign(const Tensor &t, const_reference x);
  static Tensor isgreater(const Tensor &t, const_reference x);
  static Tensor isgreaterequal(const Tensor &t, const_reference x);
  static Tensor isless(const Tensor &t, const_reference x);
  static Tensor islessequal(const Tensor &t, const_reference x);
  static Tensor islessgreater(const Tensor &t, const_reference x);
  static Tensor isunordered(const Tensor &t, const_reference x);
  static Tensor fill(const Tensor &t, const_reference x);

  // Element-wise operations with another tensor
  virtual const Tensor& add(const Tensor &x) const;
  virtual const Tensor& sub(const Tensor &x) const;
  virtual const Tensor& mul(const Tensor &x) const;
  virtual const Tensor& div(const Tensor &x) const;
  virtual const Tensor& fmod(const Tensor &x) const;
  virtual const Tensor& remainder(const Tensor &x) const;
  virtual const Tensor& fmax(const Tensor &x) const;
  virtual const Tensor& fmin(const Tensor &x) const;
  virtual const Tensor& fdim(const Tensor &x) const;
  virtual const Tensor& pow(const Tensor &x) const;
  virtual const Tensor& hypot(const Tensor &x) const;
  virtual const Tensor& atan2(const Tensor &x) const;
  virtual const Tensor& scalBN(const Tensor &x) const;
  virtual const Tensor& scaleBLN(const Tensor &x) const;
  virtual const Tensor& nextAfter(const Tensor &x) const;
  virtual const Tensor& nextToward(const Tensor &x) const;
  virtual const Tensor& copysign(const Tensor &x) const;
  virtual const Tensor& isgreater(const Tensor &x) const;
  virtual const Tensor& isgreaterequal(const Tensor &x) const;
  virtual const Tensor& isless(const Tensor &x) const;
  virtual const Tensor& islessequal(const Tensor &x) const;
  virtual const Tensor& islessgreater(const Tensor &x) const;
  virtual const Tensor& isunordered(const Tensor &x) const;

  // Non-const element-wise operations are delegated using const_cast
  virtual Tensor& add(const Tensor &x);
  virtual Tensor& sub(const Tensor &x);
  virtual Tensor& mul(const Tensor &x);
  virtual Tensor& div(const Tensor &x);
  virtual Tensor& fmod(const Tensor &x);
  virtual Tensor& remainder(const Tensor &x);
  virtual Tensor& fmax(const Tensor &x);
  virtual Tensor& fmin(const Tensor &x);
  virtual Tensor& fdim(const Tensor &x);
  virtual Tensor& pow(const Tensor &x);
  virtual Tensor& hypot(const Tensor &x);
  virtual Tensor& atan2(const Tensor &x);
  virtual Tensor& scalbn(const Tensor &x);
  virtual Tensor& scalbln(const Tensor &x);
  virtual Tensor& nextafter(const Tensor &x);
  virtual Tensor& nexttoward(const Tensor &x);
  virtual Tensor& copysign(const Tensor &x);
  virtual Tensor& isgreater(const Tensor &x);
  virtual Tensor& isgreaterequal(const Tensor &x);
  virtual Tensor& isless(const Tensor &x);
  virtual Tensor& islessgqual(const Tensor &x);
  virtual Tensor& islessgreater(const Tensor &x);
  virtual Tensor& isunordered(const Tensor &x);

  // Static element-wise operations with another tensor are delegated
  static Tensor add(const Tensor &t, const Tensor &x);
  static Tensor sub(const Tensor &t, const Tensor &x);
  static Tensor mul(const Tensor &t, const Tensor &x);
  static Tensor div(const Tensor &t, const Tensor &x);
  static Tensor fmod(const Tensor &t, const Tensor &x);
  static Tensor remainder(const Tensor &t, const Tensor &x);
  static Tensor fmax(const Tensor &t, const Tensor &x);
  static Tensor fmin(const Tensor &t, const Tensor &x);
  static Tensor fdim(const Tensor &t, const Tensor &x);
  static Tensor pow(const Tensor &t, const Tensor &x);
  static Tensor hypot(const Tensor &t, const Tensor &x);
  static Tensor atan2(const Tensor &t, const Tensor &x);
  static Tensor scalbn(const Tensor &t, const Tensor &x);
  static Tensor scalebln(const Tensor &t, const Tensor &x);
  static Tensor nextafter(const Tensor &t, const Tensor &x);
  static Tensor nexttoward(const Tensor &t, const Tensor &x);
  static Tensor copysign(const Tensor &t, const Tensor &x);
  static Tensor isgreater(const Tensor &t, const Tensor &x);
  static Tensor isgreaterequal(const Tensor &t, const Tensor &x);
  static Tensor isless(const Tensor &t, const Tensor &x);
  static Tensor islessequal(const Tensor &t, const Tensor &x);
  static Tensor islessgreater(const Tensor &t, const Tensor &x);
  static Tensor isunordered(const Tensor &t, const Tensor &x);

  // Fused operations with values and tensors
  const Tensor& fma(const_reference y, const_reference z) const;
  const Tensor& fma(const Tensor& y, const_reference z) const;
  const Tensor& fma(const Tensor& y, const Tensor& z) const;

  // Non-const fused operations are delegated using const_cast
  Tensor& fma(const_reference y, const_reference z);
  Tensor& fma(const Tensor &y, const_reference z);
  Tensor& fma(const Tensor &y, const Tensor& z);

  // Static fused operations are delegated
  static Tensor fma(const Tensor &x, const_reference y, const_reference z);
  static Tensor fma(const Tensor &x, const Tensor &y, const_reference z);
  static Tensor fma(const Tensor &x, const Tensor &y, const Tensor &z);

  // Constructor functions that can only be static
  static Tensor eye(size_type n);
  static Tensor eye(size_type m, size_type n);
  static Tensor eye(const size_storage &size);
  static Tensor linSpace(const_reference x, const_reference y, size_type n);
  static Tensor logSpace(const_reference x, const_reference y, size_type n);
  static Tensor ones(size_type n);
  static Tensor ones(size_type m, size_type n);
  static Tensor ones(size_type n0, size_type n1, size_type n2);
  static Tensor ones(size_type n0, size_type n1, size_type n2, size_type n3);
  static Tensor ones(const size_storage &size);
  static Tensor range(const_reference x, const_reference y, reference step);
  static Tensor zeros(size_type n);
  static Tensor zeros(size_type m, size_type n);
  static Tensor zeros(size_type n0, size_type n1, size_type n2);
  static Tensor zeros(size_type n0, size_type n1, size_type n2, size_type n3);
  static Tensor zeros(const size_storage &size);

 private:
  size_storage size_;
  stride_storage stride_;
  storage_pointer storage_;
  size_type offset_;
};

}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_HPP
