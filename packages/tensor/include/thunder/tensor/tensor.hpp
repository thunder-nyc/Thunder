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
#include <memory>
#include <utility>

#include "thunder/storage.hpp"

namespace thunder {
namespace tensor {

template < typename S = DoubleStorage >
class Tensor;

template < typename S >
Tensor< S > operator+(
    typename Tensor< S >::const_reference value, const Tensor< S > &x);
template < typename S >
Tensor< S > operator-(
    typename Tensor< S >::const_reference value, const Tensor< S > &x);
template < typename S >
Tensor< S > operator==(
    typename Tensor< S >::const_reference value, const Tensor< S > &x);
template < typename S >
Tensor< S > operator!=(
    typename Tensor< S >::const_reference value, const Tensor< S > &x);
template < typename S >
Tensor< S > operator>(
    typename Tensor< S >::const_reference value, const Tensor< S > &x);
template < typename S >
Tensor< S > operator<(
    typename Tensor< S >::const_reference value, const Tensor< S > &x);
template < typename S >
Tensor< S > operator>=(
    typename Tensor< S >::const_reference value, const Tensor< S > &x);
template < typename S >
Tensor< S > operator<=(
    typename Tensor< S >::const_reference value, const Tensor< S > &x);

template < typename S >
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

  // Templated conversion constructors
  template < typename Other_S >
  explicit Tensor(const Tensor< Other_S > &y);

  // Destructor
  ~Tensor();

  // Non-templated queries
  template < typename T >
  bool isSameSizeAs(const T &y) const;

  // Static non-templated queries are delegated
  template < typename T >
  static bool isSameSizeAs(const Tensor &x, const T &y);

  // Property queries
  dim_type dimension() const;
  size_storage size() const;
  size_type size(dim_type dim) const;
  size_type length() const;
  stride_storage stride() const;
  difference_type stride(dim_type dim) const;
  storage_pointer storage() const;
  size_type offset() const;
  pointer data() const;
  reference get() const;
  reference get(size_type pos) const;
  reference get(size_type pos0, size_type pos1) const;
  reference get(size_type pos0, size_type pos1, size_type pos2) const;
  reference get(size_type pos0, size_type pos1, size_type pos2,
                        size_type pos3) const;
  reference get(const size_storage &pos) const;
  bool isContiguous() const;
  bool partialContiguity(dim_type a, dim_type b) const;
  bool isUnique() const;

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
  static bool partialContiguity(const Tensor &x, dim_type a, dim_type b);
  static bool isUnique(const Tensor &x);

  // Assignment operators
  Tensor& operator=(Tensor y);

  // Paranthesis operators points to a reference of value
  reference operator()() const;
  reference operator()(size_type pos) const;
  reference operator()(size_type pos0, size_type pos1) const;
  reference operator()(size_type pos0, size_type pos1,
                               size_type pos2) const;
  reference operator()(size_type pos0, size_type pos1,
                               size_type pos2, size_type pos3) const;
  reference operator()(const size_storage &pos) const;

  // Index operators
  Tensor operator[](size_type pos) const;
  Tensor operator[](const size_storage& pos) const;
  Tensor operator[](
      const Storage< ::std::pair< size_type, size_type > > &range) const;

  // Iterators and their functions. Subtensor and value iterators.
  class iterator;
  iterator begin() const;
  iterator end() const;

  // Static iterator functions are delegated
  static iterator begin(const Tensor &x);
  static iterator end(const Tensor &x);

  // Reference iterators
  class reference_iterator;
  reference_iterator reference_begin() const;
  reference_iterator reference_end() const;

  // Static reference iterator functions are delegated
  static reference_iterator reference_begin(const Tensor &x);
  static reference_iterator reference_end(const Tensor &x);

  // Templated modifiers
  template < typename T >
  Tensor& resizeAs(const T &y);
  template < typename T >
  Tensor& sort(dim_type d = 0, bool r = false, T *y = nullptr);

  template < typename T >
  static Tensor& resizeAs(Tensor *x, const T &y);
  template < typename T >
  static Tensor& sort(
      Tensor *x, dim_type d = 0, bool r = false, T *y = nullptr);

  // Normal modifiers
  Tensor& set(const Tensor &y);
  Tensor& set(storage_pointer s, size_type os = 0);
  Tensor& set(size_storage sz, storage_pointer s, size_type os = 0);
  Tensor& set(size_storage sz, stride_storage st, storage_pointer s,
                      size_type os = 0);
  Tensor& resize(size_storage sz);
  Tensor& resize(size_type sz);
  Tensor& resize(size_type sz0, size_type sz1);
  Tensor& resize(size_type sz0, size_type sz1, size_type sz2);
  Tensor& resize(size_type sz0, size_type sz1, size_type sz2,
                         size_type sz3);
  Tensor& resize(size_storage sz, stride_storage st);
  Tensor& contiguous();
  Tensor& squeeze();
  Tensor& unique();

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

  // Templated subtensor extractors. Specialization because of type collision.
  template < typename T >
  Tensor viewAs(const T &y, size_type os = 0) const;
  Tensor viewAs(const Tensor &y, size_type os = 0) const;
  template < typename T >
  Tensor viewAs(const T &y, stride_storage st, size_type os = 0) const;
  Tensor viewAs(const Tensor &y, stride_storage st, size_type os = 0) const;
  template < typename T >
  Tensor extract(const T &y) const;
  template < typename T >
  Tensor shuffle(const T &y) const;
  template < typename T >
  Tensor permute(const T &y, dim_type d = 0) const;
  template < typename TR >
  TR getReal() const;
  template < typename TR >
  TR getImag() const;
  template < typename TR >
  TR getArg() const;
  template < typename TR >
  TR getCnrm() const;


  // Static templated subtensor extractors are delegated
  template < typename T >
  static Tensor viewAs(const Tensor &x, const T &y, size_type os = 0);
  static Tensor viewAs(const Tensor &x, const Tensor &y, size_type os = 0);
  template < typename T >
  static Tensor viewAs(const Tensor &x, const T &y, stride_storage st,
                       size_type os = 0);
  static Tensor viewAs(const Tensor &x, const Tensor &y, stride_storage st,
                       size_type os = 0);
  template < typename T >
  static Tensor extract(const Tensor &x, const T &y);
  template < typename T >
  static Tensor shuffle(const Tensor &x, const T &y);
  template < typename TR >
  static TR getReal(const Tensor &x);
  template < typename TR >
  static TR getImag(const Tensor &x);
  template < typename TR >
  static TR getArg(const Tensor &x);
  template < typename TR >
  static TR getCnrm(const Tensor &x);

  // Extract subtensors or transformations -- no need for non-const overload
  Tensor narrow(dim_type dim, size_type pos, size_type size) const;
  Tensor select(dim_type dim, size_type pos) const;
  Tensor view(size_type sz0) const;
  Tensor view(size_type sz0, size_type sz1) const;
  Tensor view(size_type sz0, size_type sz1, size_type sz2) const;
  Tensor view(size_type sz0, size_type sz1, size_type sz2,
                      size_type sz3) const;
  Tensor view(size_storage sz, size_type os = 0) const;
  Tensor view(size_storage sz, stride_storage st,
                      size_type os = 0) const;
  Tensor transpose(dim_type dim0 = 0, dim_type dim1 = 1) const;
  Tensor unfold(dim_type dim, size_type size, size_type step) const;
  Tensor clone() const;
  Tensor cat(const Tensor &y, dim_type dim = 0) const;
  Tensor reshape(size_type sz0) const;
  Tensor reshape(size_type sz0, size_type sz1) const;
  Tensor reshape(size_type sz0, size_type sz1, size_type sz2) const;
  Tensor reshape(size_type sz0, size_type sz1, size_type sz2,
                         size_type sz3) const;
  Tensor reshape(size_storage sz) const;

  // Static subtensor or transformation extractors are delegated
  static Tensor narrow(const Tensor &x, dim_type dim, size_type pos,
                       size_type size);
  static Tensor select(const Tensor &x, dim_type dim, size_type pos);
  static Tensor view(const Tensor &x, size_type sz0);
  static Tensor view(const Tensor &x, size_type sz0, size_type sz1);
  static Tensor view(const Tensor &x, size_type sz0, size_type sz1,
                     size_type sz2);
  static Tensor view(const Tensor &x, size_type sz0, size_type sz1,
                     size_type sz2, size_type sz3);
  static Tensor view(const Tensor &x, size_storage sz, size_type os = 0);
  static Tensor view(const Tensor &x, size_storage sz, stride_storage st,
                     size_type os = 0);
  static Tensor transpose(const Tensor &x, dim_type dim0 = 0,
                          dim_type dim1 = 1);
  static Tensor unfold(const Tensor &x, dim_type dim, size_type size,
                       size_type step);
  static Tensor clone(const Tensor& t);
  static Tensor cat(const Tensor &x, const Tensor &y, dim_type dim = 0);
  static Tensor reshape(const Tensor &x, size_type sz0);
  static Tensor reshape(const Tensor &x, size_type sz0, size_type sz1);
  static Tensor reshape(const Tensor &x, size_type sz0, size_type sz1,
                        size_type sz2);
  static Tensor reshape(const Tensor &x, size_type sz0, size_type sz1,
                        size_type sz2, size_type sz3);
  static Tensor reshape(const Tensor &x, size_storage sz);

  // Type conversions
  template < typename T >
  T type() const;

  // Static type conversions are delegated
  template < typename T >
  static T type(const Tensor& x);


  // lambda applications
  const Tensor& apply(
      const ::std::function< value_type(value_type) > &lambda) const;
  const Tensor& apply(
      const ::std::function< value_type(const value_type&) > &lambda) const;
  const Tensor& apply(
      const ::std::function< void(value_type&) > &lambda) const;
  const Tensor& apply(
      const ::std::function< void(value_type*) > &lambda) const;

  // Non-const lambda applications are delegated using const_cast
  Tensor& apply(
      const ::std::function< value_type(value_type) > &lambda);
  Tensor& apply(
      const ::std::function< value_type(const value_type&) > &lambda);
  Tensor& apply(const ::std::function< void(value_type&) > &lambda);
  Tensor& apply(const ::std::function< void(value_type*) > &lambda);

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

  // Element-wise mathematical operations that are free of parameters
  const Tensor& abs() const;
  const Tensor& fabs() const;
  const Tensor& exp() const;
  const Tensor& exp2() const;
  const Tensor& expm1() const;
  const Tensor& log() const;
  const Tensor& log10() const;
  const Tensor& log2() const;
  const Tensor& log1p() const;
  const Tensor& sqrt() const;
  const Tensor& cbrt() const;
  const Tensor& sin() const;
  const Tensor& cos() const;
  const Tensor& tan() const;
  const Tensor& asin() const;
  const Tensor& acos() const;
  const Tensor& atan() const;
  const Tensor& sinh() const;
  const Tensor& cosh() const;
  const Tensor& tanh() const;
  const Tensor& asinh() const;
  const Tensor& acosh() const;
  const Tensor& atanh() const;
  const Tensor& erf() const;
  const Tensor& erfc() const;
  const Tensor& tgamma() const;
  const Tensor& lgamma() const;
  const Tensor& ceil() const;
  const Tensor& floor() const;
  const Tensor& trunc() const;
  const Tensor& round() const;
  const Tensor& nearbyint() const;
  const Tensor& rint() const;
  const Tensor& logb() const;
  const Tensor& fpclassify() const;
  const Tensor& isfinite() const;
  const Tensor& isinf() const;
  const Tensor& isnan() const;
  const Tensor& isnormal() const;
  const Tensor& signbit() const;
  const Tensor& zero() const;
  const Tensor& real() const;
  const Tensor& imag() const;
  const Tensor& arg() const;
  const Tensor& cnrm() const;
  const Tensor& conj() const;
  const Tensor& proj() const;

  // Non-const element-wise operations are delegated using const_cast
  Tensor& abs();
  Tensor& fabs();
  Tensor& exp();
  Tensor& exp2();
  Tensor& expm1();
  Tensor& log();
  Tensor& log10();
  Tensor& log2();
  Tensor& log1p();
  Tensor& sqrt();
  Tensor& cbrt();
  Tensor& sin();
  Tensor& cos();
  Tensor& tan();
  Tensor& asin();
  Tensor& acos();
  Tensor& atan();
  Tensor& sinh();
  Tensor& cosh();
  Tensor& tanh();
  Tensor& asinh();
  Tensor& acosh();
  Tensor& atanh();
  Tensor& erf();
  Tensor& erfc();
  Tensor& tgamma();
  Tensor& lgamma();
  Tensor& ceil();
  Tensor& floor();
  Tensor& trunc();
  Tensor& round();
  Tensor& nearbyint();
  Tensor& rint();
  Tensor& logb();
  Tensor& fpclassify();
  Tensor& isfinite();
  Tensor& isinf();
  Tensor& isnan();
  Tensor& isnormal();
  Tensor& signbit();
  Tensor& zero();
  Tensor& real();
  Tensor& imag();
  Tensor& arg();
  Tensor& cnrm();
  Tensor& conj();
  Tensor& proj();

  // static element-wise mathematical operations are deligated
  static Tensor abs(const Tensor &x);
  static Tensor fabs(const Tensor &x);
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
  const Tensor& add(const_reference y) const;
  const Tensor& sub(const_reference y) const;
  const Tensor& mul(const_reference y) const;
  const Tensor& div(const_reference y) const;
  const Tensor& fmod(const_reference y) const;
  const Tensor& remainder(const_reference y) const;
  const Tensor& fmax(const_reference y) const;
  const Tensor& fmin(const_reference y) const;
  const Tensor& fdim(const_reference y) const;
  const Tensor& pow(const_reference y) const;
  const Tensor& hypot(const_reference y) const;
  const Tensor& atan2(const_reference y) const;
  const Tensor& ldexp(const_reference y) const;
  const Tensor& scalbn(const_reference y) const;
  const Tensor& scalbln(const_reference y) const;
  const Tensor& nextafter(const_reference y) const;
  const Tensor& nexttoward(const_reference y) const;
  const Tensor& copysign(const_reference y) const;
  const Tensor& isgreater(const_reference y) const;
  const Tensor& isgreaterequal(const_reference y) const;
  const Tensor& isless(const_reference y) const;
  const Tensor& islessequal(const_reference y) const;
  const Tensor& islessgreater(const_reference y) const;
  const Tensor& isunordered(const_reference y) const;
  const Tensor& fill(const_reference y) const;

  // Non-const element-wise operations are delegated using const_cast
  Tensor& add(const_reference y);
  Tensor& sub(const_reference y);
  Tensor& mul(const_reference y);
  Tensor& div(const_reference y);
  Tensor& fmod(const_reference y);
  Tensor& remainder(const_reference y);
  Tensor& fmax(const_reference y);
  Tensor& fmin(const_reference y);
  Tensor& fdim(const_reference y);
  Tensor& pow(const_reference y);
  Tensor& hypot(const_reference y);
  Tensor& atan2(const_reference y);
  Tensor& ldexp(const_reference y);
  Tensor& scalbn(const_reference y);
  Tensor& scalbln(const_reference y);
  Tensor& nextafter(const_reference y);
  Tensor& nexttoward(const_reference y);
  Tensor& copysign(const_reference y);
  Tensor& isgreater(const_reference y);
  Tensor& isgreaterequal(const_reference y);
  Tensor& isless(const_reference y);
  Tensor& islessequal(const_reference y);
  Tensor& islessgreater(const_reference y);
  Tensor& isunordered(const_reference y);
  Tensor& fill(const_reference y);

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
  static Tensor ldexp(const Tensor &x, const_reference y);
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

  // Templated element-wise operations with another tensor
  template < typename T >
  const Tensor& copy(const T &y) const;
  template < typename T >
  Tensor& copy(const T& y);

  // Element-wise operations with another tensor
  const Tensor& add(const Tensor &y) const;
  const Tensor& sub(const Tensor &y) const;
  const Tensor& mul(const Tensor &y) const;
  const Tensor& div(const Tensor &y) const;
  const Tensor& fmod(const Tensor &y) const;
  const Tensor& remainder(const Tensor &y) const;
  const Tensor& fmax(const Tensor &y) const;
  const Tensor& fmin(const Tensor &y) const;
  const Tensor& fdim(const Tensor &y) const;
  const Tensor& pow(const Tensor &y) const;
  const Tensor& hypot(const Tensor &y) const;
  const Tensor& atan2(const Tensor &y) const;
  const Tensor& ldexp(const Tensor &y) const;
  const Tensor& scalbn(const Tensor &y) const;
  const Tensor& scalbln(const Tensor &y) const;
  const Tensor& nextafter(const Tensor &y) const;
  const Tensor& nexttoward(const Tensor &y) const;
  const Tensor& copysign(const Tensor &y) const;
  const Tensor& isgreater(const Tensor &y) const;
  const Tensor& isgreaterequal(const Tensor &y) const;
  const Tensor& isless(const Tensor &y) const;
  const Tensor& islessequal(const Tensor &y) const;
  const Tensor& islessgreater(const Tensor &y) const;
  const Tensor& isunordered(const Tensor &y) const;

  // Non-const element-wise operations are delegated using const_cast
  Tensor& add(const Tensor &y);
  Tensor& sub(const Tensor &y);
  Tensor& mul(const Tensor &y);
  Tensor& div(const Tensor &y);
  Tensor& fmod(const Tensor &y);
  Tensor& remainder(const Tensor &y);
  Tensor& fmax(const Tensor &y);
  Tensor& fmin(const Tensor &y);
  Tensor& fdim(const Tensor &y);
  Tensor& pow(const Tensor &y);
  Tensor& hypot(const Tensor &y);
  Tensor& atan2(const Tensor &y);
  Tensor& ldexp(const Tensor &y);
  Tensor& scalbn(const Tensor &y);
  Tensor& scalbln(const Tensor &y);
  Tensor& nextafter(const Tensor &y);
  Tensor& nexttoward(const Tensor &y);
  Tensor& copysign(const Tensor &y);
  Tensor& isgreater(const Tensor &y);
  Tensor& isgreaterequal(const Tensor &y);
  Tensor& isless(const Tensor &y);
  Tensor& islessequal(const Tensor &y);
  Tensor& islessgreater(const Tensor &y);
  Tensor& isunordered(const Tensor &y);

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
  static Tensor ldexp(const Tensor &x, const Tensor &y);
  static Tensor scalbn(const Tensor &x, const Tensor &y);
  static Tensor scalbln(const Tensor &x, const Tensor &y);
  static Tensor nextafter(const Tensor &x, const Tensor &y);
  static Tensor nexttoward(const Tensor &x, const Tensor &y);
  static Tensor copysign(const Tensor &x, const Tensor &y);
  static Tensor isgreater(const Tensor &x, const Tensor &y);
  static Tensor isgreaterequal(const Tensor &x, const Tensor &y);
  static Tensor isless(const Tensor &x, const Tensor &y);
  static Tensor islessequal(const Tensor &x, const Tensor &y);
  static Tensor islessgreater(const Tensor &x, const Tensor &y);
  static Tensor isunordered(const Tensor &x, const Tensor &y);

  // Templated ternary functions
  template < typename TR >
  const Tensor& polar(typename TR::const_reference r, const TR& theta) const;
  template < typename TR >
  const Tensor& polar(const TR& r, typename TR::const_reference theta) const;
  template < typename TR >
  const Tensor& polar(const TR& r, const TR& theta) const;

  // Non-const templated ternary functions are delegated using const_cast
  template < typename TR >
  Tensor& polar(typename TR::const_reference r, const TR& theta);
  template < typename TR >
  Tensor& polar(const TR& r, typename TR::const_reference theta);
  template < typename TR >
  Tensor& polar(const TR& r, const TR& theta);

  // Static templated ternary functions are delegated
  template < typename TR >
  static Tensor polar(
      const Tensor &x, typename TR::const_reference r, const TR& theta);
  template < typename TR >
  static Tensor polar(
      const Tensor &x, const TR& r, typename TR::const_reference theta);
  template < typename TR >
  static Tensor polar(
      const Tensor &x, const TR& r, const TR& theta);

  // Ternary operations with values and tensors
  const Tensor& polar(const_reference y, const_reference z) const;
  const Tensor& polar(const Tensor& y, const_reference z) const;
  const Tensor& polar(const_reference y, const Tensor& z) const;
  const Tensor& polar(const Tensor& y, const Tensor& z) const;
  const Tensor& fma(const_reference y, const_reference z) const;
  const Tensor& fma(const Tensor& y, const_reference z) const;
  const Tensor& fma(const_reference y, const Tensor& z) const;
  const Tensor& fma(const Tensor& y, const Tensor& z) const;

  // Non-const ternary operations are delegated using const_cast
  Tensor& polar(const_reference y, const_reference z);
  Tensor& polar(const Tensor &y, const_reference z);
  Tensor& polar(const_reference y, const Tensor& z);
  Tensor& polar(const Tensor &y, const Tensor& z);
  Tensor& fma(const_reference y, const_reference z);
  Tensor& fma(const Tensor &y, const_reference z);
  Tensor& fma(const_reference y, const Tensor& z);
  Tensor& fma(const Tensor &y, const Tensor& z);

  // Static ternary operations are delegated
  static Tensor polar(const Tensor &x, const_reference y, const_reference z);
  static Tensor polar(const Tensor &x, const Tensor &y, const_reference z);
  static Tensor polar(const Tensor &x, const_reference y, const Tensor &z);
  static Tensor polar(const Tensor &x, const Tensor &y, const Tensor &z);
  static Tensor fma(const Tensor &x, const_reference y, const_reference z);
  static Tensor fma(const Tensor &x, const Tensor &y, const_reference z);
  static Tensor fma(const Tensor &x, const_reference y, const Tensor &z);
  static Tensor fma(const Tensor &x, const Tensor &y, const Tensor &z);

  // Reduction operations
  value_type max(Tensor< size_storage > *pos) const;
  value_type min(Tensor< size_storage > *pos) const;
  value_type max() const;
  value_type min() const;
  value_type sum() const;
  value_type prod() const;
  value_type mean() const;
  value_type var() const;
  value_type std() const;

  // Static reduction operations are deligated
  static value_type max(const Tensor &x, Tensor< size_storage > *pos);
  static value_type min(const Tensor &x, Tensor< size_storage > *pos);
  static value_type max(const Tensor &x);
  static value_type min(const Tensor &x);
  static value_type sum(const Tensor &x);
  static value_type prod(const Tensor &x);
  static value_type mean(const Tensor &x);
  static value_type var(const Tensor &x);
  static value_type std(const Tensor &x);

  // Reduction operations along a dimension
  Tensor max(dim_type d, Tensor< size_storage > *pos) const;
  Tensor min(dim_type d, Tensor< size_storage > *pos) const;
  Tensor max(dim_type d) const;
  Tensor min(dim_type d) const;
  Tensor sum(dim_type d) const;
  Tensor prod(dim_type d) const;
  Tensor mean(dim_type d) const;
  Tensor var(dim_type d) const;
  Tensor std(dim_type d) const;

  // Static reduction operations are deligated
  static Tensor max(const Tensor &x, dim_type d, Tensor< size_storage > *pos);
  static Tensor min(const Tensor &x, dim_type d, Tensor< size_storage > *pos);
  static Tensor max(const Tensor &x, dim_type d);
  static Tensor min(const Tensor &x, dim_type d);
  static Tensor sum(const Tensor &x, dim_type d);
  static Tensor prod(const Tensor &x, dim_type d);
  static Tensor mean(const Tensor &x, dim_type d);
  static Tensor var(const Tensor &x, dim_type d);
  static Tensor std(const Tensor &x, dim_type d);

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

  // Templated constructor functions
  template < typename TR >
  static Tensor polars(typename TR::const_reference r, const TR& theta);
  template < typename TR >
  static Tensor polars(const TR& r, typename TR::const_reference theta);
  template < typename TR >
  static Tensor polars(const TR& r, const TR& theta);

  // Arithmetic operators with value are delegated
  Tensor operator+(const_reference value) const;
  Tensor operator-(const_reference value) const;
  template < typename FS >
  friend Tensor< FS > operator+(
      typename Tensor< FS >::const_reference value, const Tensor< FS > &x);
  template < typename FS >
  friend Tensor< FS > operator-(
      typename Tensor< FS >::const_reference value, const Tensor< FS > &x);
  const Tensor& operator+=(const_reference value) const;
  const Tensor& operator-=(const_reference value) const;
  Tensor& operator+=(const_reference value);
  Tensor& operator-=(const_reference value);

  // Arithmetic operators are delegated
  Tensor operator+(const Tensor &y) const;
  Tensor operator-(const Tensor &y) const;
  const Tensor& operator+=(const Tensor &y) const;
  const Tensor& operator-=(const Tensor &y) const;
  Tensor& operator+=(const Tensor &y);
  Tensor& operator-=(const Tensor &y);

  // Comparison operators with value are delegated
  Tensor operator==(const_reference value) const;
  Tensor operator!=(const_reference value) const;
  Tensor operator>(const_reference value) const;
  Tensor operator<(const_reference value) const;
  Tensor operator>=(const_reference value) const;
  Tensor operator<=(const_reference value) const;
  template < typename FS >
  friend Tensor< FS > operator==(
      typename Tensor< FS >::const_reference value, const Tensor< FS > &x);
  template < typename FS >
  friend Tensor< FS > operator!=(
      typename Tensor< FS >::const_reference value, const Tensor< FS > &x);
  template < typename FS >
  friend Tensor< FS > operator>(
      typename Tensor< FS >::const_reference value, const Tensor< FS > &x);
  template < typename FS >
  friend Tensor< FS > operator<(
      typename Tensor< FS >::const_reference value, const Tensor< FS > &x);
  template < typename FS >
  friend Tensor< FS > operator>=(
      typename Tensor< FS >::const_reference value, const Tensor< FS > &x);
  template < typename FS >
  friend Tensor< FS > operator<=(
      typename Tensor< FS >::const_reference value, const Tensor< FS > &x);

  // Comparison operators with another tensor are delegated
  Tensor operator==(const Tensor &y) const;
  Tensor operator!=(const Tensor &y) const;
  Tensor operator>(const Tensor &y) const;
  Tensor operator<(const Tensor &y) const;
  Tensor operator>=(const Tensor &y) const;
  Tensor operator<=(const Tensor &y) const;

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

  size_type position() const;

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

  size_storage position() const;

 protected:
  const Tensor *tensor_;
  size_storage position_;
};

}  // namespace tensor
}  // namespace thunder

namespace thunder {
namespace serializer {

template < typename C, typename S >
void save(C *s, const ::thunder::tensor::Tensor< S > &t);

template < typename C, typename S >
void load(C *s, ::thunder::tensor::Tensor< S > *t);

}  // namespace serializer
}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_HPP_
