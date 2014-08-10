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

#include "thunder/storage.hpp"

namespace thunder{

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
  typedef size_storage::size_type dim_type;

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
  virtual Tensor &operator=(Tensor other);
  virtual Tensor &operator=(const_reference value);
  friend reference operator=(reference value, const Tensor& tensor);

  // Index operators
  virtual Tensor operator[](size_type pos) const;
  virtual Tensor operator[](const size_storage &pos) const;
  virtual Tensor operator[](size_type pos_a, size_type pos_b) const;
  virtual Tensor operator[](const size_storage &pos_a,
                            const size_storage &pos_b) const;

  // Iterators and their functions. Subtensor and value iterators.
  class iterator;
  class const_iterator;
  class value_iterator;
  iterator begin();
  iterator end();
  const_iterator begin() const;
  const_iterator end() const;
  value_iterator value_begin() const;
  value_iterator value_end() const;

  // Property queries
  dim_type Dimension() const;
  size_storage Size() const;
  size_type Size(dim_type dim) const;
  size_type Count() const;
  stride_storage Stride() const;
  difference_type Stride(dim_type dim) const;
  storage_pointer Storage() const;
  size_type Offset() const;
  pointer Data() const;
  bool IsContiguous() const;
  template < typename Other_S >
  bool IsSameSizeAs (const Tensor< Other_S > &other) const;

  // Storage setters
  Tensor& Set(const Tensor &other);
  Tensor& Set(const storage_pointer &storage, size_type offset = 0);
  Tensor& Set(const size_storage &size, const storage_pointer &storage,
              size_type offset = 0);
  Tensor& Set(const size_storage &size, const stride_storage &stride,
              const stroage_pointer &storage, size_type offset = 0);

  // Copy data from another tensor with same number of elements
  template < typename Other_S >
  Tensor& Copy(const Tensor< Other_S > &other);

  // Resize Operations
  Tensor& Resize(const size_storage &size);
  Tensor& Resize(const size_storage &size, const stride_storage &stride);
  template < typename Other_S >
  Tensor& ResizeAs(const Tensor< Other_S > &other);

  // Extract subtensors or transformations
  Tensor Narrow(dim_type dim, size_type pos, size_type size) const;
  Tensor Select(dim_type dim, size_type pos) const;
  Tensor View(const size_storage &size) const;
  template < typename Other_S >
  Tensor ViewAs(const Tensor< Other_S > &other);
  Tensor Transpose();
  Tensor Transpose(dim_type dim0, dim_type dim1);
  Tensor Unfold(dim_type dim, size_type size, size_type step);

  // Apply a lambda. Returns reference to *this if it is a modifier.
  virtual Tensor& Apply(::std::function< value_type(value_type) > &lambda);
  virtual Tensor& Apply(::std::function< value_type(const value_type&) > &lambda);
  virtual Tensor& Apply(::std::function< void(value_type&) > &lambda);
  virtual Tensor& Apply(::std::function< void(value_type*) > &lambda);
  virtual void Apply(::std::function< void(value_type) > &lambda) const;
  virtual void Apply(::std::function< void(const value_type&) > &lambda) const;
  virtual void Apply(::std::function< void(const value_type*) > &lambda) const;

  // Constructor functions in which this stores the result.
  virtual Tensor& Cat(const Tensor& s, const Tensor& t, dim_type dim);
  virtual Tensor& Diag(const Tensor& s);
  virtual Tensor& Eye(size_type n);
  virtual Tensor& Eye(size_type m, size_type n);
  virtual Tensor& Eye(const size_storage &size);
  virtual Tensor& LinSpace(const_reference x, const_reference y, size_type n);
  virtual Tensor& LogSpace(const_reference x, const_reference y, size_type n);
  virtual Tensor& Ones(size_type n);
  virtual Tensor& Ones(size_type m, size_type n);
  virtual Tensor& Ones(size_type n0, size_type n1, size_type n2);
  virtual Tensor& Ones(size_type n0, size_type n1, size_type n2, size_type n3);
  virtual Tensor& Ones(const size_storage &size);
  virtual Tensor& Range(const_reference x, const_reference y,
                        const_reference step);
  virtual Tensor& Reshape(const Tensor& tensor, const size_storage size);
  virtual Tensor& TriL(const Tensor &s);
  virtual Tensor& TriU(const Tensor &s);
  virtual Tensor& Zeros(size_type n);
  virtual Tensor& Zeros(size_type m, size_type n);
  virtual Tensor& Zeros(size_type n0, size_type n1, size_type n2);
  virtual Tensor& Zeros(size_type n0, size_type n1, size_type n2, size_type n3);
  virtual Tensor& Zeros(const size_storage &size);

  // All random generators are either double or int. Static casts are used.
  virtual Tensor& Rand();
  virtual Tensor& Uniform(double a = 0.0, double b = 1.0);
  virtual Tensor& UniformReal(double a = 0.0, double b = 1.0);
  virtual Tensor& UniformInt(int a = 0,
                             int b = ::std::numeric_limits<IntType>::max())
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

  // Element-wise mathematical operations on this that are in standard library
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
  virtual Tensor& Contiguous();

  // Reduction operations
  virtual value_type Max();
  virtual value_type Min();
  virtual value_type Sum();
  virtual value_type Mean();
  virtual value_type Var();
  virtual value_type Std();

  // Element-wise operations with a constant
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

  // Element-wise operations with another tensor
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

  // Static constructor functions are delegated
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

  // Static random generators are deligated
  static Tensor Rand(const size_storage &size);
  static Tensor Uniform(const size_storage &size, double a = 0.0,
                        double b = 1.0);
  static Tensor UniformReal(const size_storage &size, double a = 0.0,
                            double b = 1.0);
  static Tensor UniformInt(const size_storage &size, int a = 0,
                           int b = ::std::numeric_limits<IntType>::max());
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
  static Tensor Contiguous();

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

 private:
  size_storage size_;
  stride_storage stride_;
  storage_pointer storage_;
  size_type offset_;
};

}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_HPP
