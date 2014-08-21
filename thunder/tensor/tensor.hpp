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

#ifndef THUNDER_TENSOR_TENSOR_HPP_
#define THUNDER_TENSOR_TENSOR_HPP_

#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "thunder/storage.hpp"

namespace thunder {
namespace tensor {

template < typename S = DoubleStorage >
class Tensor {
 public:
  // Typedefs from storage
  typedef S storage_type;
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
  explicit Tensor();
  explicit Tensor(size_storage sz);
  explicit Tensor(size_type sz0);
  Tensor(size_type sz0, size_type sz1);
  Tensor(size_type sz0, size_type sz1, size_type sz2);
  Tensor(size_type sz0, size_type sz1, size_type sz2, size_type sz3);
  explicit Tensor(storage_pointer s, size_type os = 0);
  Tensor(size_storage sz, storage_pointer s, size_type os = 0);
  Tensor(size_storage sz, stride_storage st);
  Tensor(size_storage sz, stride_storage st, storage_pointer s,
         size_type os = 0);
  Tensor(const Tensor &y);
  Tensor(Tensor &&y);

  // Destructor
  virtual ~Tensor();

  // Non-virtual templated queries
  template < typename T >
  bool isSameSizeAs(const T &y) const;

  // Static non-virtual templated queries are delegated
  template < typename T >
  static bool isSameSizeAs(const Tensor &x, const T &y);

  // Property queries
  virtual dim_type dimension() const;
  virtual size_storage size() const;
  virtual size_type size(dim_type dim) const;
  virtual size_type length() const;
  virtual stride_storage stride() const;
  virtual difference_type stride(dim_type dim) const;
  virtual storage_pointer storage() const;
  virtual size_type offset() const;
  virtual pointer data() const;
  virtual reference get() const;
  virtual reference get(size_type pos) const;
  virtual reference get(size_type pos0, size_type pos1) const;
  virtual reference get(size_type pos0, size_type pos1, size_type pos2) const;
  virtual reference get(size_type pos0, size_type pos1, size_type pos2,
                        size_type pos3) const;
  virtual reference get(const size_storage &pos) const;
  virtual bool isContiguous() const;

  // Static property queries are delegated
  static dim_type dimension(const Tensor &x);
  static size_storage size(const Tensor &x);
  static size_type size(const Tensor &x, dim_type dim);
  static size_type length(const Tensor &x);
  static stride_storage stride(const Tensor &x);
  static difference_type stride(const Tensor &x, dim_type dim);
  static storage_pointer storage(const Tensor &x);
  static size_type offset(const Tensor &x);
  static pointer data(const Tensor &x);
  static reference get(const Tensor &x);
  static reference get(const Tensor &x, size_type pos);
  static reference get(const Tensor &x, size_type pos0, size_type pos1);
  static reference get(const Tensor &x, size_type pos0, size_type pos1,
                       size_type pos2);
  static reference get(const Tensor &x, size_type pos0, size_type pos1,
                       size_type pos2, size_type pos3);
  static reference get(const Tensor &x, const size_storage &pos);
  static bool isContiguous(const Tensor &x);

  // Assignment operators
  virtual Tensor& operator=(Tensor y);

  // Paranthesis operators points to a reference of value
  virtual reference operator()() const;
  virtual reference operator()(size_type pos) const;
  virtual reference operator()(size_type pos0, size_type pos1) const;
  virtual reference operator()(size_type pos0, size_type pos1,
                               size_type pos2) const;
  virtual reference operator()(size_type pos0, size_type pos1,
                               size_type pos2, size_type pos3) const;
  virtual reference operator()(const size_storage &pos) const;

  // Index operators
  virtual Tensor operator[](size_type pos) const;
  virtual Tensor operator[](const size_storage& pos) const;
  virtual Tensor operator[](
      const Storage< ::std::pair< size_type, size_type > > &range) const;

  // Iterators and their functions. Subtensor and value iterators.
  class iterator;
  virtual iterator begin() const;
  virtual iterator end() const;

  // Static iterator functions are delegated
  static iterator begin(const Tensor &x);
  static iterator end(const Tensor &x);

  // Reference iterators
  class reference_iterator;
  virtual reference_iterator reference_begin() const;
  virtual reference_iterator reference_end() const;

  // Static reference iterator functions are delegated
  static reference_iterator reference_begin(const Tensor &x);
  static reference_iterator reference_end(const Tensor &x);

  // Non-virtual templated modifiers
  template < typename T >
  Tensor& copy(const T &y);
  template < typename T >
  Tensor& resizeAs(const T &y);

  // Static non-virtual templated modifiers are delegated
  template < typename T >
  static Tensor& copy(Tensor *x, const T &y);
  template < typename T >
  static Tensor& resizeAs(Tensor *x, const T &y);

  // Normal modifiers
  virtual Tensor& set(const Tensor &y);
  virtual Tensor& set(storage_pointer s, size_type os = 0);
  virtual Tensor& set(size_storage sz, storage_pointer s, size_type os = 0);
  virtual Tensor& set(size_storage sz, stride_storage st, storage_pointer s,
                      size_type os = 0);
  virtual Tensor& resize(size_storage sz);
  virtual Tensor& resize(size_type sz);
  virtual Tensor& resize(size_type sz0, size_type sz1);
  virtual Tensor& resize(size_type sz0, size_type sz1, size_type sz2);
  virtual Tensor& resize(size_type sz0, size_type sz1, size_type sz2,
                         size_type sz3);
  virtual Tensor& resize(size_storage sz, stride_storage st);
  virtual Tensor& contiguous();
  virtual Tensor& squeeze();
  virtual Tensor& unique();

  // Static modifiers are delegated
  static Tensor& set(Tensor *x, const Tensor &y);
  static Tensor& set(Tensor *x, storage_pointer s, size_type os = 0);
  static Tensor& set(Tensor *x, size_storage sz, storage_pointer s,
                     size_type os = 0);
  static Tensor& set(Tensor *x, size_storage sz, stride_storage st,
                     storage_pointer s, size_type os = 0);
  static Tensor& resize(Tensor *x, size_storage sz);
  static Tensor& resize(Tensor *x, size_type sz);
  static Tensor& resize(Tensor *x, size_type sz0, size_type sz1);
  static Tensor& resize(Tensor *x, size_type sz0, size_type sz1, size_type sz2);
  static Tensor& resize(Tensor *x, size_type sz0, size_type sz1, size_type sz2,
                        size_type sz3);
  static Tensor& resize(Tensor *x, size_storage sz, stride_storage st);
  static Tensor& contiguous(Tensor *x);
  static Tensor& squeeze(Tensor *x);
  static Tensor& unique(Tensor *x);

  // Templated subtensor extractors
  template < typename T >
  Tensor viewAs(const T &y, size_type os = 0) const;
  template < typename T >
  Tensor viewAs(const T &y, const stride_storage &st,
                size_type os = 0) const;
  template < typename T >
  Tensor extract(const T &y) const;
  template < typename T >
  Tensor shuffle(const T &y) const;


  // Static templated subtensor extractors are delegated
  template < typename T >
  static Tensor viewAs(const Tensor &x, const T &y, size_type os = 0);
  template < typename T >
  static Tensor viewAs(const Tensor &x, const T &y,
                       const stride_storage &st, size_type os = 0);
  template < typename T >
  static Tensor extract(const Tensor &x, const T &y);
  template < typename T >
  static Tensor shuffle(const Tensor &x, const T &y);

  /* !!!! THIS IS MARK FOR NOT IMPLEMENTED YET !!!!

  // Extract subtensors or transformations -- no need for non-const overload
  virtual Tensor narrow(dim_type dim, size_type pos, size_type size) const;
  virtual Tensor select(dim_type dim, size_type pos) const;
  virtual Tensor view(size_type sz0, size_type os = 0) const;
  virtual Tensor view(size_type sz0, size_type sz1, size_type os = 0) const;
  virtual Tensor view(size_type sz0, size_type sz1, size_type sz2,
                      size_type os = 0) const;
  virtual Tensor view(size_type sz0, size_type sz1, size_type sz2,
                      size_type sz3, size_type os = 0) const;
  virtual Tensor view(const size_storage &sz, size_type os = 0) const;
  virtual Tensor view(const size_storage &sz, const stride_storage &st,
                      size_type os = 0) const;
  virtual Tensor transpose(dim_type dim0 = 0, dim_type dim1 = 1) const;
  virtual Tensor unfold(dim_type dim, size_type size, size_type step) const;
  virtual Tensor clone() const;
  virtual Tensor cat(const Tensor &y, dim_type dim) const;
  virtual Tensor reshape(size_type sz0) const;
  virtual Tensor reshape(size_type sz0, size_type sz1) const;
  virtual Tensor reshape(size_type sz0, size_type sz1, size_type sz2) const;
  virtual Tensor reshape(size_type sz0, size_type sz1, size_type sz2,
                         size_type sz3) const;
  virtual Tensor reshape(const size_storage size) const;

  // Static subtensor or transformation extractors are delegated
  static Tensor narrow(const Tensor &x, dim_type dim, size_type pos,
                       size_type size);
  static Tensor select(const Tensor &x, dim_type dim, size_type pos);
  static Tensor view(const Tensor &x, size_type sz0);
  static Tensor view(const Tensor &x, size_type sz0, size_type sz1);
  static Tensor view(const Tensor &x, size_type sz0, size_type sz1,
                     size_type size2);
  static Tensor view(const Tensor &x, size_type sz0, size_type sz1,
                     size_type sz2, size_type sz3);
  static Tensor view(const Tensor &x, const size_storage &sz);
  static Tensor view(const Tensor &x, const size_storage &sz,
                     const stride_storage &st, size_type os = 0);
  static Tensor transpose(const Tensor &t, dim_type dim0 = 0,
                          dim_type dim1 = 1);
  static Tensor unfold(const Tensor &t, dim_type dim, size_type size,
                       size_type step);
  static Tensor clone(const Tensor& t);
  static Tensor cat(const Tensor &x, const Tensor &y, dim_type dim);
  static Tensor reshape(const Tensor &x, size_type s0);
  static Tensor reshape(const Tensor &x, size_type s0, size_type s1);
  static Tensor reshape(const Tensor &x, size_type s0, size_type s1,
                        size_type s2);
  static Tensor reshape(const Tensor &x, size_type s0, size_type s1,
                        size_type s2, size_type s3);
  static Tensor reshape(const Tensor &x, const size_storage size);

  // Type conversions
  template < typename T >
  T type();
  virtual const Tensor& type() const;
  virtual Tensor& type();

  // Static type conversions are delegated
  template < typename T >
  static T type(const Tensor& t);
  static Tensor& type(Tensor &x);
  static const Tensor& type(const Tensor &x);

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
  static Tensor apply(const Tensor& x,
                      const ::std::function< value_type(value_type) > &lambda);
  static Tensor apply(
      const Tensor &x,
      const ::std::function< value_type(const value_type&) > &lambda);
  static Tensor apply(const Tensor &x,
                      const ::std::function< void(value_type&) > &lambda);
  static Tensor apply(const Tensor &x,
                      const ::std::function< void(value_type*) > &lambda);

  // Reduction operations
  virtual value_type max(Tensor< size_type > *pos = nullptr) const;
  virtual value_type min(Tensor< size_type > *pos = nullptr) const;
  virtual value_type sum() const;
  virtual value_type prod() const;
  virtual value_type mean() const;
  virtual value_type var() const;
  virtual value_type std() const;

  // Static reduction operations are deligated
  static value_type max(const Tensor &x, Tensor< size_type > *pos = nullptr);
  static value_type min(const Tensor &x, Tensor< size_type > *pos = nullptr);
  static value_type sum(const Tensor &x);
  static value_type prod(const Tensor &x);
  static value_type mean(const Tensor &x);
  static value_type var(const Tensor &x);
  static value_type std(const Tensor &x);

  // Reduction operations along a dimension
  virtual value_type max(dim_type d, Tensor< size_type > *pos = nullptr) const;
  virtual value_type min(dim_type d, Tensor< size_type > *pos = nullptr) const;
  virtual value_type sum(dim_type d) const;
  virtual value_type prod(dim_type d) const;
  virtual value_type mean(dim_type d) const;
  virtual value_type var(dim_type d) const;
  virtual value_type std(dim_type d) const;

  // Static reduction operations are deligated
  static value_type max(const Tensor &x, dim_type d,
                        Tensor< size_type > *pos = nullptr);
  static value_type min(const Tensor &x, dim_type d,
                        Tensor< size_type > *pos = nullptr);
  static value_type sum(const Tensor &x, dim_type d);
  static value_type prod(const Tensor &x, dim_type d);
  static value_type mean(const Tensor &x, dim_type d);
  static value_type var(const Tensor &x, dim_type d);
  static value_type std(const Tensor &x, dim_type d);

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
  template < typename RF = double >
  static Tensor uniform(const size_storage &sz, RF a = 0.0, RF b = 1.0);
  template < typename RF = double >
  static Tensor uniformReal(const size_storage &sz, RF a = 0.0, RF b = 1.0);
  template < typename RI = int >
  static Tensor uniformInt(const size_storage &sz, RI a = 0,
                           RI b = ::std::numeric_limits< RI >::max());
  template < typename RF = double >
  static Tensor bernoulli(const size_storage &sz, RF p = 0.5);
  template < typename RI = int, typename RF = double >
  static Tensor ninomial(const size_storage &sz, RI t = 1, RF p = 0.5);
  template < typename RI = int, typename RF = double >
  static Tensor negativeBinomial(const size_storage &sz, RI k = 1,
                                 RF p = 0.5);
  template < typename RF = double >
  static Tensor geometric(const size_storage &sz, RF p = 0.5);
  template < typename RF = double >
  static Tensor poisson(const size_storage &sz, RF mean = 1.0);
  template < typename RF = double >
  static Tensor exponential(const size_storage &sz, RF lambda = 1.0);
  template < typename RF = double >
  static Tensor gamma(const size_storage &sz, RF alpha = 1.0, RF beta = 1.0);
  template < typename RF = double >
  static Tensor weibull(const size_storage &sz, RF a = 1.0, RF b = 1.0);
  template < typename RF = double >
  static Tensor extremeValue(const size_storage &sz, RF a = 0.0, RF b = 1.0);
  template < typename RF = double >
  static Tensor normal(const size_storage &sz, RF mean = 0.0,
                       RF stddev = 1.0);
  template < typename RF = double >
  static Tensor logNormal(const size_storage &sz, RF m = 0.0, RF s = 1.0);
  template < typename RF = double >
  static Tensor chiSquared(const size_storage &sz, RF n = 1.0);
  template < typename RF = double >
  static Tensor cauchy(const size_storage &sz, RF a = 0.0, RF b = 1.0);
  template < typename RF = double >
  static Tensor fisherF(const size_storage &sz, RF m = 1.0, RF n = 1.0);
  template < typename RF = double >
  static Tensor studentT(const size_storage &sz, RF n = 1.0);

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
  virtual const Tensor& real() const;
  virtual const Tensor& imag() const;
  virtual const Tensor& arg() const;
  virtual const Tensor& cnrm() const;
  virtual const Tensor& conj() const;
  virtual const Tensor& proj() const;

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
  virtual Tensor& real();
  virtual Tensor& imag();
  virtual Tensor& arg();
  virtual Tensor& cnrm();
  virtual Tensor& conj();
  virtual Tensor& proj();

  // static element-wise mathematical operations are deligated
  static Tensor exp(const Tensor &x);
  static Tensor exp2(const Tensor &x);
  static Tensor expm1(const Tensor &x);
  static Tensor log(const Tensor &x);
  static Tensor log10(const Tensor &x);
  static Tensor log2(const Tensor &x);
  static Tensor log1p(const Tensor &x);
  static Tensor sqrt(const Tensor &x);
  static Tensor cbrt(const Tensor &x);
  static Tensor sin(const Tensor &x);
  static Tensor cos(const Tensor &x);
  static Tensor tan(const Tensor &x);
  static Tensor asin(const Tensor &x);
  static Tensor acos(const Tensor &x);
  static Tensor atan(const Tensor &x);
  static Tensor sinh(const Tensor &x);
  static Tensor cosh(const Tensor &x);
  static Tensor tanh(const Tensor &x);
  static Tensor asinh(const Tensor &x);
  static Tensor acosh(const Tensor &x);
  static Tensor atanh(const Tensor &x);
  static Tensor erf(const Tensor &x);
  static Tensor erfc(const Tensor &x);
  static Tensor tgamma(const Tensor &x);
  static Tensor lgamma(const Tensor &x);
  static Tensor ceil(const Tensor &x);
  static Tensor floor(const Tensor &x);
  static Tensor trunc(const Tensor &x);
  static Tensor round(const Tensor &x);
  static Tensor nearbyint(const Tensor &x);
  static Tensor rint(const Tensor &x);
  static Tensor logb(const Tensor &x);
  static Tensor fpclassify(const Tensor &x);
  static Tensor isfinite(const Tensor &x);
  static Tensor isinf(const Tensor &x);
  static Tensor isnan(const Tensor &x);
  static Tensor isnormal(const Tensor &x);
  static Tensor signbit(const Tensor &x);
  static Tensor zero(const Tensor &x);
  static Tensor real(const Tensor &x);
  static Tensor imag(const Tensor &x);
  static Tensor arg(const Tensor &x);
  static Tensor cnrm(const Tensor &x);
  static Tensor conj(const Tensor &x);
  static Tensor proj(const Tensor &x);

  // Element-wise operations with a value
  virtual const Tensor& add(const_reference y) const;
  virtual const Tensor& sub(const_reference y) const;
  virtual const Tensor& mul(const_reference y) const;
  virtual const Tensor& div(const_reference y) const;
  virtual const Tensor& fmod(const_reference y) const;
  virtual const Tensor& remainder(const_reference y) const;
  virtual const Tensor& fmax(const_reference y) const;
  virtual const Tensor& fmin(const_reference y) const;
  virtual const Tensor& fdim(const_reference y) const;
  virtual const Tensor& pow(const_reference y) const;
  virtual const Tensor& hypot(const_reference y) const;
  virtual const Tensor& atan2(const_reference y) const;
  virtual const Tensor& ldexp(const_reference y) const;
  virtual const Tensor& Scalbn(const_reference y) const;
  virtual const Tensor& Scalbln(const_reference y) const;
  virtual const Tensor& nextafter(const_reference y) const;
  virtual const Tensor& nexttoward(const_reference y) const;
  virtual const Tensor& copysign(const_reference y) const;
  virtual const Tensor& isgreater(const_reference y) const;
  virtual const Tensor& isgreaterequal(const_reference y) const;
  virtual const Tensor& isless(const_reference y) const;
  virtual const Tensor& islessequal(const_reference y) const;
  virtual const Tensor& islessgreater(const_reference y) const;
  virtual const Tensor& isunordered(const_reference y) const;
  virtual const Tensor& fill(const_reference y) const;

  // Non-const element-wise operations are delegated using static_cast
  virtual Tensor& add(const_reference y);
  virtual Tensor& sub(const_reference y);
  virtual Tensor& mul(const_reference y);
  virtual Tensor& div(const_reference y);
  virtual Tensor& fmod(const_reference y);
  virtual Tensor& remainder(const_reference y);
  virtual Tensor& fmax(const_reference y);
  virtual Tensor& fmin(const_reference y);
  virtual Tensor& fdim(const_reference y);
  virtual Tensor& pow(const_reference y);
  virtual Tensor& hypot(const_reference y);
  virtual Tensor& atan2(const_reference y);
  virtual Tensor& ldexp(const_reference y);
  virtual Tensor& scalbn(const_reference y);
  virtual Tensor& scalbln(const_reference y);
  virtual Tensor& nextafter(const_reference y);
  virtual Tensor& nexttoward(const_reference y);
  virtual Tensor& copysign(const_reference y);
  virtual Tensor& isgreater(const_reference y);
  virtual Tensor& isgreaterequal(const_reference y);
  virtual Tensor& isless(const_reference y);
  virtual Tensor& islessequal(const_reference y);
  virtual Tensor& islessgreater(const_reference y);
  virtual Tensor& isunordered(const_reference y);
  virtual Tensor& fill(const_reference y);

  // Static element-wise operations with a constant are delegated
  static Tensor add(const Tensor &x, const_reference y);
  static Tensor sub(const Tensor &x, const_reference y);
  static Tensor mul(const Tensor &x, const_reference y);
  static Tensor div(const Tensor &x, const_reference y);
  static Tensor fmod(const Tensor &x, const_reference y);
  static Tensor remainder(const Tensor &x, const_reference y);
  static Tensor fmax(const Tensor &x, const_reference y);
  static Tensor fmin(const Tensor &x, const_reference y);
  static Tensor fdim(const Tensor &x, const_reference y);
  static Tensor pow(const Tensor &x, const_reference y);
  static Tensor hypot(const Tensor &x, const_reference y);
  static Tensor atan2(const Tensor &x, const_reference y);
  static Tensor scalbn(const Tensor &x, const_reference y);
  static Tensor scalbln(const Tensor &x, const_reference y);
  static Tensor nextafter(const Tensor &x, const_reference y);
  static Tensor nexttoward(const Tensor &x, const_reference y);
  static Tensor copysign(const Tensor &x, const_reference y);
  static Tensor isgreater(const Tensor &x, const_reference y);
  static Tensor isgreaterequal(const Tensor &x, const_reference y);
  static Tensor isless(const Tensor &x, const_reference y);
  static Tensor islessequal(const Tensor &x, const_reference y);
  static Tensor islessgreater(const Tensor &x, const_reference y);
  static Tensor isunordered(const Tensor &x, const_reference y);
  static Tensor fill(const Tensor &x, const_reference y);

  // Element-wise operations with another tensor
  virtual const Tensor& add(const Tensor &y) const;
  virtual const Tensor& sub(const Tensor &y) const;
  virtual const Tensor& mul(const Tensor &y) const;
  virtual const Tensor& div(const Tensor &y) const;
  virtual const Tensor& fmod(const Tensor &y) const;
  virtual const Tensor& remainder(const Tensor &y) const;
  virtual const Tensor& fmax(const Tensor &y) const;
  virtual const Tensor& fmin(const Tensor &y) const;
  virtual const Tensor& fdim(const Tensor &y) const;
  virtual const Tensor& pow(const Tensor &y) const;
  virtual const Tensor& hypot(const Tensor &y) const;
  virtual const Tensor& atan2(const Tensor &y) const;
  virtual const Tensor& scalbn(const Tensor &y) const;
  virtual const Tensor& scalbln(const Tensor &y) const;
  virtual const Tensor& nextafter(const Tensor &y) const;
  virtual const Tensor& nexttoward(const Tensor &y) const;
  virtual const Tensor& copysign(const Tensor &y) const;
  virtual const Tensor& isgreater(const Tensor &y) const;
  virtual const Tensor& isgreaterequal(const Tensor &y) const;
  virtual const Tensor& isless(const Tensor &y) const;
  virtual const Tensor& islessequal(const Tensor &y) const;
  virtual const Tensor& islessgreater(const Tensor &y) const;
  virtual const Tensor& isunordered(const Tensor &y) const;

  // Non-const element-wise operations are delegated using const_cast
  virtual Tensor& add(const Tensor &y);
  virtual Tensor& sub(const Tensor &y);
  virtual Tensor& mul(const Tensor &y);
  virtual Tensor& div(const Tensor &y);
  virtual Tensor& fmod(const Tensor &y);
  virtual Tensor& remainder(const Tensor &y);
  virtual Tensor& fmax(const Tensor &y);
  virtual Tensor& fmin(const Tensor &y);
  virtual Tensor& fdim(const Tensor &y);
  virtual Tensor& pow(const Tensor &y);
  virtual Tensor& hypot(const Tensor &y);
  virtual Tensor& atan2(const Tensor &y);
  virtual Tensor& scalbn(const Tensor &y);
  virtual Tensor& scalbln(const Tensor &y);
  virtual Tensor& nextafter(const Tensor &y);
  virtual Tensor& nexttoward(const Tensor &y);
  virtual Tensor& copysign(const Tensor &y);
  virtual Tensor& isgreater(const Tensor &y);
  virtual Tensor& isgreaterequal(const Tensor &y);
  virtual Tensor& isless(const Tensor &y);
  virtual Tensor& islessgqual(const Tensor &y);
  virtual Tensor& islessgreater(const Tensor &y);
  virtual Tensor& isunordered(const Tensor &y);

  // Static element-wise operations with another tensor are delegated
  static Tensor add(const Tensor &x, const Tensor &y);
  static Tensor sub(const Tensor &x, const Tensor &y);
  static Tensor mul(const Tensor &x, const Tensor &y);
  static Tensor div(const Tensor &x, const Tensor &y);
  static Tensor fmod(const Tensor &x, const Tensor &y);
  static Tensor remainder(const Tensor &x, const Tensor &y);
  static Tensor fmax(const Tensor &x, const Tensor &y);
  static Tensor fmin(const Tensor &x, const Tensor &y);
  static Tensor fdim(const Tensor &x, const Tensor &y);
  static Tensor pow(const Tensor &x, const Tensor &y);
  static Tensor hypot(const Tensor &x, const Tensor &y);
  static Tensor atan2(const Tensor &x, const Tensor &y);
  static Tensor scalbn(const Tensor &x, const Tensor &y);
  static Tensor scalebln(const Tensor &x, const Tensor &y);
  static Tensor nextafter(const Tensor &x, const Tensor &y);
  static Tensor nexttoward(const Tensor &x, const Tensor &y);
  static Tensor copysign(const Tensor &x, const Tensor &y);
  static Tensor isgreater(const Tensor &x, const Tensor &y);
  static Tensor isgreaterequal(const Tensor &x, const Tensor &y);
  static Tensor isless(const Tensor &x, const Tensor &y);
  static Tensor islessequal(const Tensor &x, const Tensor &y);
  static Tensor islessgreater(const Tensor &x, const Tensor &y);
  static Tensor isunordered(const Tensor &x, const Tensor &y);

  // Fused operations with values and tensors
  const Tensor& fma(const_reference y, const_reference z) const;
  const Tensor& fma(const Tensor& y, const_reference z) const;
  const Tensor& fma(const_reference y, const Tensor& z) const;
  const Tensor& fma(const Tensor& y, const Tensor& z) const;

  // Non-const fused operations are delegated using const_cast
  Tensor& fma(const_reference y, const_reference z);
  Tensor& fma(const Tensor &y, const_reference z);
  Tensor& fma(const_reference y, const Tensor& z);
  Tensor& fma(const Tensor &y, const Tensor& z);

  // Static fused operations are delegated
  static Tensor fma(const Tensor &x, const_reference y, const_reference z);
  static Tensor fma(const Tensor &x, const Tensor &y, const_reference z);
  static Tensor fma(const Tensor &x, const_reference y, const Tensor &z);
  static Tensor fma(const Tensor &x, const Tensor &y, const Tensor &z);

  // Constructor functions that can only be static
  static Tensor ones(size_type n);
  static Tensor ones(size_type m, size_type n);
  static Tensor ones(size_type n0, size_type n1, size_type n2);
  static Tensor ones(size_type n0, size_type n1, size_type n2, size_type n3);
  static Tensor ones(const size_storage &sz);
  static Tensor zeros(size_type n);
  static Tensor zeros(size_type m, size_type n);
  static Tensor zeros(size_type n0, size_type n1, size_type n2);
  static Tensor zeros(size_type n0, size_type n1, size_type n2, size_type n3);
  static Tensor zeros(const size_storage &sz);

  // Templated constructor functions can only be static
  template < typename TR >
  static Tensor polar(const_reference r, const TR& theta);
  template < typename TR >
  static Tensor polar(const TR& r, reference theta);
  template < typename TR >
  static Tensor polar(const TR& r, const TR& theta);

  // Arithmetic operators with value are delegated
  virtual Tensor operator+(const_reference value) const;
  virtual Tensor operator-(const_reference value) const;
  friend Tensor operator+(const_reference value, const Tensor &x);
  friend Tensor operator-(const_reference value, const Tensor &x);
  virtual Tensor& operator+=(const_reference value) const;
  virtual Tensor& operator-=(const_reference value) const;

  // Arithmetic operators are delegated
  virtual Tensor operator+(const Tensor &y) const;
  virtual Tensor operator-(const Tensor &y) const;
  virtual Tensor& operator+=(const Tensor &y) const;
  virtual Tensor& operator-=(const Tensor &y) const;

  // Comparison operators with value are delegated
  virtual Tensor operator==(const_reference value) const;
  virtual Tensor operator>(const_reference value) const;
  virtual Tensor operator<(const_reference value) const;
  virtual Tensor operator<=(const_reference value) const;
  virtual Tensor operator>=(const_reference value) const;
  friend Tensor operator==(const_reference value, const Tensor &x);
  friend Tensor operator>(const_reference value, const Tensor &x);
  friend Tensor operator<(const_reference value, const Tensor &x);
  friend Tensor operator>=(const_reference value, const Tensor &x);
  friend Tensor operator<=(const_reference value, const Tensor &x);

  // Comparison operators with another tensor are delegated
  virtual Tensor operator==(const Tensor &y) const;
  virtual Tensor operator>(const Tensor &y) const;
  virtual Tensor operator<(const Tensor &y) const;
  virtual Tensor operator<=(const Tensor &y) const;
  virtual Tensor operator>=(const Tensor &y) const;

  !!! THIS IS MARK FOR NOT IMPLEMENTED YET !!! */

 protected:
  size_storage size_;
  stride_storage stride_;
  storage_pointer storage_;
  size_type offset_;
};

template < typename S >
class Tensor< S >::iterator {
 public:
  typedef ::std::input_iterator_tag iterator_category;

  explicit iterator(const Tensor &x, size_type pos = 0);
  iterator(const iterator& it);
  iterator(iterator&& it);
  ~iterator();

  iterator& operator=(iterator it);

  bool operator==(const iterator& it) const;
  bool operator!=(const iterator& it) const;

  iterator& operator++();
  iterator operator++(int);

  const Tensor& operator*() const;
  const Tensor* operator->() const;

 protected:
  const Tensor *tensor_;
  size_type position_;
  Tensor current_;
};

template < typename S >
class Tensor< S >::reference_iterator {
 public:
  typedef std::input_iterator_tag iterator_category;

  explicit reference_iterator(const Tensor &x);
  reference_iterator(const Tensor &x, size_storage pos);
  reference_iterator(const reference_iterator& it);
  reference_iterator(reference_iterator&& it);
  ~reference_iterator();

  reference_iterator& operator=(reference_iterator it);

  bool operator==(const reference_iterator& it) const;
  bool operator!=(const reference_iterator& it) const;

  reference_iterator& operator++();
  reference_iterator operator++(int);

  reference operator*() const;
  pointer operator->() const;

 protected:
  const Tensor *tensor_;
  size_storage position_;
};

}  // namespace tensor
}  // namespace thunder

#include "thunder/tensor/tensor-inl.hpp"

#endif  // THUNDER_TENSOR_TENSOR_HPP_
