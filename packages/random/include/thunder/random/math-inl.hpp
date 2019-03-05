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

#ifndef THUNDER_RANDOM_MATH_INL_HPP_
#define THUNDER_RANDOM_MATH_INL_HPP_

#include "thunder/random/math.hpp"

#include <random>

#include "thunder/exception.hpp"

namespace thunder {
namespace random {
namespace math {

template < typename R >
const typename R::tensor_type& random(
    R *r, const typename R::tensor_type &t, typename R::integer_type a,
    typename R::integer_type b) {
  typedef typename R::tensor_type T;
  typedef typename R::integer_type I;
  ::std::uniform_int_distribution< I > distribution(a, b);
  if (t.partialContiguity(0, t.dimension() - 1)) {
    typename T::pointer t_pointer = t.data();
    typename T::size_type t_length = t.length();
    typename T::difference_type t_step = t.stride(t.dimension() - 1);
    for (typename T::size_type i = 0; i < t_length; ++i) {
      t_pointer[i * t_step] = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  } else {
    for (typename T::reference_iterator t_begin = t.reference_begin(),
             t_end = t.reference_end(); t_begin != t_end; ++t_begin) {
      *t_begin = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  }
  return t;
}

template < typename R >
const typename R::tensor_type& uniform(
    R *r, const typename R::tensor_type &t, typename R::float_type a,
    typename R::float_type b) {
  typedef typename R::tensor_type T;
  typedef typename R::float_type F;
  ::std::uniform_real_distribution< F > distribution(a, b);
  if (t.partialContiguity(0, t.dimension() - 1)) {
    typename T::pointer t_pointer = t.data();
    typename T::size_type t_length = t.length();
    typename T::difference_type t_step = t.stride(t.dimension() - 1);
    for (typename T::size_type i = 0; i < t_length; ++i) {
      t_pointer[i * t_step] = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  } else {
    for (typename T::reference_iterator t_begin = t.reference_begin(),
             t_end = t.reference_end(); t_begin != t_end; ++t_begin) {
      *t_begin = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  }
  return t;
}

template < typename R >
const typename R::tensor_type& bernoulli(
    R *r, const typename R::tensor_type &t, typename R::float_type p) {
  typedef typename R::tensor_type T;
  ::std::bernoulli_distribution distribution(p);
  if (t.partialContiguity(0, t.dimension() - 1)) {
    typename T::pointer t_pointer = t.data();
    typename T::size_type t_length = t.length();
    typename T::difference_type t_step = t.stride(t.dimension() - 1);
    for (typename T::size_type i = 0; i < t_length; ++i) {
      t_pointer[i * t_step] = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  } else {
    for (typename T::reference_iterator t_begin = t.reference_begin(),
             t_end = t.reference_end(); t_begin != t_end; ++t_begin) {
      *t_begin = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  }
  return t;
}

template < typename R >
const typename R::tensor_type& binomial(
    R *r, const typename R::tensor_type &t, typename R::integer_type s,
    typename R::float_type p) {
  typedef typename R::tensor_type T;
  typedef typename R::integer_type I;
  ::std::binomial_distribution< I > distribution(s, p);
  if (t.partialContiguity(0, t.dimension() - 1)) {
    typename T::pointer t_pointer = t.data();
    typename T::size_type t_length = t.length();
    typename T::difference_type t_step = t.stride(t.dimension() - 1);
    for (typename T::size_type i = 0; i < t_length; ++i) {
      t_pointer[i * t_step] = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  } else {
    for (typename T::reference_iterator t_begin = t.reference_begin(),
             t_end = t.reference_end(); t_begin != t_end; ++t_begin) {
      *t_begin = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  }
  return t;
}

template < typename R >
const typename R::tensor_type& negativeBinomial(
    R *r, const typename R::tensor_type &t, typename R::integer_type k,
    typename R::float_type p) {
  typedef typename R::tensor_type T;
  typedef typename R::integer_type I;
  ::std::negative_binomial_distribution< I > distribution(k, p);
  if (t.partialContiguity(0, t.dimension() - 1)) {
    typename T::pointer t_pointer = t.data();
    typename T::size_type t_length = t.length();
    typename T::difference_type t_step = t.stride(t.dimension() - 1);
    for (typename T::size_type i = 0; i < t_length; ++i) {
      t_pointer[i * t_step] = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  } else {
    for (typename T::reference_iterator t_begin = t.reference_begin(),
             t_end = t.reference_end(); t_begin != t_end; ++t_begin) {
      *t_begin = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  }
  return t;
}

template < typename R >
const typename R::tensor_type& geometric(
    R *r, const typename R::tensor_type &t, typename R::float_type p) {
  typedef typename R::tensor_type T;
  typedef typename R::integer_type I;
  ::std::geometric_distribution< I > distribution(p);
  if (t.partialContiguity(0, t.dimension() - 1)) {
    typename T::pointer t_pointer = t.data();
    typename T::size_type t_length = t.length();
    typename T::difference_type t_step = t.stride(t.dimension() - 1);
    for (typename T::size_type i = 0; i < t_length; ++i) {
      t_pointer[i * t_step] = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  } else {
    for (typename T::reference_iterator t_begin = t.reference_begin(),
             t_end = t.reference_end(); t_begin != t_end; ++t_begin) {
      *t_begin = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  }
  return t;
}

template < typename R >
const typename R::tensor_type& poisson(
    R *r, const typename R::tensor_type &t, typename R::float_type mean) {
  typedef typename R::tensor_type T;
  typedef typename R::integer_type I;
  ::std::poisson_distribution< I > distribution(mean);
  if (t.partialContiguity(0, t.dimension() - 1)) {
    typename T::pointer t_pointer = t.data();
    typename T::size_type t_length = t.length();
    typename T::difference_type t_step = t.stride(t.dimension() - 1);
    for (typename T::size_type i = 0; i < t_length; ++i) {
      t_pointer[i * t_step] = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  } else {
    for (typename T::reference_iterator t_begin = t.reference_begin(),
             t_end = t.reference_end(); t_begin != t_end; ++t_begin) {
      *t_begin = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  }
  return t;
}

template < typename R >
const typename R::tensor_type& exponential(
    R *r, const typename R::tensor_type &t, typename R::float_type lambda) {
  typedef typename R::tensor_type T;
  typedef typename R::float_type F;
  ::std::exponential_distribution< F > distribution(lambda);
  if (t.partialContiguity(0, t.dimension() - 1)) {
    typename T::pointer t_pointer = t.data();
    typename T::size_type t_length = t.length();
    typename T::difference_type t_step = t.stride(t.dimension() - 1);
    for (typename T::size_type i = 0; i < t_length; ++i) {
      t_pointer[i * t_step] = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  } else {
    for (typename T::reference_iterator t_begin = t.reference_begin(),
             t_end = t.reference_end(); t_begin != t_end; ++t_begin) {
      *t_begin = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  }
  return t;
}

template < typename R >
const typename R::tensor_type& gamma(
    R *r, const typename R::tensor_type &t, typename R::float_type alpha,
    typename R::float_type beta) {
  typedef typename R::tensor_type T;
  typedef typename R::float_type F;
  ::std::gamma_distribution< F > distribution(alpha, beta);
  if (t.partialContiguity(0, t.dimension() - 1)) {
    typename T::pointer t_pointer = t.data();
    typename T::size_type t_length = t.length();
    typename T::difference_type t_step = t.stride(t.dimension() - 1);
    for (typename T::size_type i = 0; i < t_length; ++i) {
      t_pointer[i * t_step] = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  } else {
    for (typename T::reference_iterator t_begin = t.reference_begin(),
             t_end = t.reference_end(); t_begin != t_end; ++t_begin) {
      *t_begin = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  }
  return t;
}

template < typename R >
const typename R::tensor_type& weibull(
    R *r, const typename R::tensor_type &t, typename R::float_type a,
    typename R::float_type b) {
  typedef typename R::tensor_type T;
  typedef typename R::float_type F;
  ::std::weibull_distribution< F > distribution(a, b);
  if (t.partialContiguity(0, t.dimension() - 1)) {
    typename T::pointer t_pointer = t.data();
    typename T::size_type t_length = t.length();
    typename T::difference_type t_step = t.stride(t.dimension() - 1);
    for (typename T::size_type i = 0; i < t_length; ++i) {
      t_pointer[i * t_step] = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  } else {
    for (typename T::reference_iterator t_begin = t.reference_begin(),
             t_end = t.reference_end(); t_begin != t_end; ++t_begin) {
      *t_begin = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  }
  return t;
}

template < typename R >
const typename R::tensor_type& extremeValue(
    R *r, const typename R::tensor_type &t, typename R::float_type a,
    typename R::float_type b) {
  typedef typename R::tensor_type T;
  typedef typename R::float_type F;
  ::std::extreme_value_distribution< F > distribution(a, b);
  if (t.partialContiguity(0, t.dimension() - 1)) {
    typename T::pointer t_pointer = t.data();
    typename T::size_type t_length = t.length();
    typename T::difference_type t_step = t.stride(t.dimension() - 1);
    for (typename T::size_type i = 0; i < t_length; ++i) {
      t_pointer[i * t_step] = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  } else {
    for (typename T::reference_iterator t_begin = t.reference_begin(),
             t_end = t.reference_end(); t_begin != t_end; ++t_begin) {
      *t_begin = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  }
  return t;
}

template < typename R >
const typename R::tensor_type& normal(
    R *r, const typename R::tensor_type &t, typename R::float_type mean,
    typename R::float_type stddev) {
  typedef typename R::tensor_type T;
  typedef typename R::float_type F;
  ::std::normal_distribution< F > distribution(mean, stddev);
  if (t.partialContiguity(0, t.dimension() - 1)) {
    typename T::pointer t_pointer = t.data();
    typename T::size_type t_length = t.length();
    typename T::difference_type t_step = t.stride(t.dimension() - 1);
    for (typename T::size_type i = 0; i < t_length; ++i) {
      t_pointer[i * t_step] = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  } else {
    for (typename T::reference_iterator t_begin = t.reference_begin(),
             t_end = t.reference_end(); t_begin != t_end; ++t_begin) {
      *t_begin = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  }
  return t;
}

template < typename R >
const typename R::tensor_type& logNormal(
    R *r, const typename R::tensor_type &t, typename R::float_type m,
    typename R::float_type s) {
  typedef typename R::tensor_type T;
  typedef typename R::float_type F;
  ::std::lognormal_distribution< F > distribution(m, s);
  if (t.partialContiguity(0, t.dimension() - 1)) {
    typename T::pointer t_pointer = t.data();
    typename T::size_type t_length = t.length();
    typename T::difference_type t_step = t.stride(t.dimension() - 1);
    for (typename T::size_type i = 0; i < t_length; ++i) {
      t_pointer[i * t_step] = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  } else {
    for (typename T::reference_iterator t_begin = t.reference_begin(),
             t_end = t.reference_end(); t_begin != t_end; ++t_begin) {
      *t_begin = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  }
  return t;
}

template < typename R >
const typename R::tensor_type& chiSquared(
    R *r, const typename R::tensor_type &t, typename R::float_type n) {
  typedef typename R::tensor_type T;
  typedef typename R::float_type F;
  ::std::chi_squared_distribution< F > distribution(n);
  if (t.partialContiguity(0, t.dimension() - 1)) {
    typename T::pointer t_pointer = t.data();
    typename T::size_type t_length = t.length();
    typename T::difference_type t_step = t.stride(t.dimension() - 1);
    for (typename T::size_type i = 0; i < t_length; ++i) {
      t_pointer[i * t_step] = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  } else {
    for (typename T::reference_iterator t_begin = t.reference_begin(),
             t_end = t.reference_end(); t_begin != t_end; ++t_begin) {
      *t_begin = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  }
  return t;
}

template < typename R >
const typename R::tensor_type& cauchy(
    R *r, const typename R::tensor_type &t, typename R::float_type a,
    typename R::float_type b) {
  typedef typename R::tensor_type T;
  typedef typename R::float_type F;
  ::std::cauchy_distribution< F > distribution(a, b);
  if (t.partialContiguity(0, t.dimension() - 1)) {
    typename T::pointer t_pointer = t.data();
    typename T::size_type t_length = t.length();
    typename T::difference_type t_step = t.stride(t.dimension() - 1);
    for (typename T::size_type i = 0; i < t_length; ++i) {
      t_pointer[i * t_step] = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  } else {
    for (typename T::reference_iterator t_begin = t.reference_begin(),
             t_end = t.reference_end(); t_begin != t_end; ++t_begin) {
      *t_begin = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  }
  return t;
}

template < typename R >
const typename R::tensor_type& fisherF(
    R *r, const typename R::tensor_type &t, typename R::float_type m,
    typename R::float_type n) {
  typedef typename R::tensor_type T;
  typedef typename R::float_type F;
  ::std::fisher_f_distribution< F > distribution(m, n);
  if (t.partialContiguity(0, t.dimension() - 1)) {
    typename T::pointer t_pointer = t.data();
    typename T::size_type t_length = t.length();
    typename T::difference_type t_step = t.stride(t.dimension() - 1);
    for (typename T::size_type i = 0; i < t_length; ++i) {
      t_pointer[i * t_step] = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  } else {
    for (typename T::reference_iterator t_begin = t.reference_begin(),
             t_end = t.reference_end(); t_begin != t_end; ++t_begin) {
      *t_begin = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  }
  return t;
}

template < typename R >
const typename R::tensor_type& studentT(
    R *r, const typename R::tensor_type &t, typename R::float_type n) {
  typedef typename R::tensor_type T;
  typedef typename R::float_type F;
  ::std::student_t_distribution< F > distribution(n);
  if (t.partialContiguity(0, t.dimension() - 1)) {
    typename T::pointer t_pointer = t.data();
    typename T::size_type t_length = t.length();
    typename T::difference_type t_step = t.stride(t.dimension() - 1);
    for (typename T::size_type i = 0; i < t_length; ++i) {
      t_pointer[i * t_step] = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  } else {
    for (typename T::reference_iterator t_begin = t.reference_begin(),
             t_end = t.reference_end(); t_begin != t_end; ++t_begin) {
      *t_begin = static_cast< typename T::value_type >(
          distribution(*(r->generatorPointer())));
    }
  }
  return t;
}

template < typename R >
const typename R::tensor_type& randperm(
    R *r, const typename R::tensor_type &t) {
  typedef typename R::tensor_type T;
  if (t.dimension() != 1) {
    throw invalid_argument("Input tensor dimension exceeds 1.");
  }
  typename T::pointer t_pointer = t.data();
  typename T::size_type t_length = t.length();
  typename T::difference_type t_step = t.stride(t.dimension() - 1);
  for (typename T::size_type i = 0; i < t_length; ++i) {
    t_pointer[i * t_step] = static_cast< typename T::value_type >(i);
  }
  for (typename T::size_type i = 0; i < t_length; ++i) {
    ::std::swap(t_pointer[i * t_step],
                t_pointer[((*(r->generatorPointer()))() %
                           (t_length - i)) + i * t_step]);
  }
  return t;
}

}  // namespace math
}  // namespace random
}  // namespace thunder

#endif  // THUNDER_RANDOM_MATH_INL_HPP_
