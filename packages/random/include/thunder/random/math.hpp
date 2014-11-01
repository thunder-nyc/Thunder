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

#ifndef THUNDER_RANDOM_MATH_HPP_
#define THUNDER_RANDOM_MATH_HPP_

namespace thunder {
namespace random {
namespace math {

template < typename R >
const typename R::tensor_type& random(
    R *r, const typename R::tensor_type &t, typename R::integer_type a,
    typename R::integer_type b);

template < typename R >
const typename R::tensor_type& uniform(
    R *r, const typename R::tensor_type &t, typename R::float_type a,
    typename R::float_type b);

template < typename R >
const typename R::tensor_type& bernoulli(
    R *r, const typename R::tensor_type &t, typename R::float_type p);

template < typename R >
const typename R::tensor_type& binomial(
    R *r, const typename R::tensor_type &t, typename R::integer_type s,
    typename R::float_type p);

template < typename R >
const typename R::tensor_type& negativeBinomial(
    R *r, const typename R::tensor_type &t, typename R::integer_type k,
    typename R::float_type p);

template < typename R >
const typename R::tensor_type& geometric(
    R *r, const typename R::tensor_type &t, typename R::float_type p);

template < typename R >
const typename R::tensor_type& poisson(
    R *r, const typename R::tensor_type &t, typename R::float_type mean);

template < typename R >
const typename R::tensor_type& exponential(
    R *r, const typename R::tensor_type &t, typename R::float_type lambda);

template < typename R >
const typename R::tensor_type& gamma(
    R *r, const typename R::tensor_type &t, typename R::float_type alpha,
    typename R::float_type beta);

template < typename R >
const typename R::tensor_type& weibull(
    R *r, const typename R::tensor_type &t, typename R::float_type a,
    typename R::float_type b);

template < typename R >
const typename R::tensor_type& extremeValue(
    R *r, const typename R::tensor_type &t, typename R::float_type a,
    typename R::float_type b);

template < typename R >
const typename R::tensor_type& normal(
    R *r, const typename R::tensor_type &t, typename R::float_type mean,
    typename R::float_type stddev);

template < typename R >
const typename R::tensor_type& logNormal(
    R *r, const typename R::tensor_type &t, typename R::float_type m,
    typename R::float_type s);

template < typename R >
const typename R::tensor_type& chiSquared(
    R *r, const typename R::tensor_type &t, typename R::float_type n);

template < typename R >
const typename R::tensor_type& cauchy(
    R *r, const typename R::tensor_type &t, typename R::float_type a,
    typename R::float_type b);

template < typename R >
const typename R::tensor_type& fisherF(
    R *r, const typename R::tensor_type &t, typename R::float_type m,
    typename R::float_type n);

template < typename R >
const typename R::tensor_type& studentT(
    R *r, const typename R::tensor_type &t, typename R::float_type n);

}  // namespace math
}  // namespace random
}  // namespace thunder

#endif  // typename R::tensor_typeHUNDER_RANDOM_MATH_HPP_
