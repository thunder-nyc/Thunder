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

#ifndef THUNDER_LINALG_LINALG_HPP_
#define THUNDER_LINALG_LINALG_HPP_

#include "thunder/tensor.hpp"

namespace thunder {
namespace linalg {

template < typename T = DoubleTensor, typename H = int >
class Linalg {
 public:
  typedef typename T::value_type value_type;
  typedef typename T::size_type size_type;
  
  Linalg();
  template < typename... G >
  Linalg(G... g);
  ~Linalg();
  
  // Const result BLAS level-1 routines
  const T& asum(const T &x, const T &r);
  const T& axpy(const value_type &a, const T &x, const T &y, const T &r);
  const T& copy(const T &x, const T &y);
  const T& dot(const T &x, const T &y, const T &r);
  const T& sdot(const T &x, const T &y, const T &r);
  const T& dotc(const T &x, const T &y, const T &r);
  const T& nrm2(const T &x, const T &r);
  const T& rot(const T &x, const T &y, const value_type &c,
               const value_type &s);
  const T& rotm(const T &x, const T &y, const T &P);
  const T& scal(const value_type &a, const T &x);
  const T& swap(const T &x, const T &y);
  const SizeTensor& iamax(const T &x, const SizeTensor &r);
  const SizeTensor& iamin(const T &x, const SizeTensor &r);
  const T& cabs1(const T &x);

  // Const result BLAS level-2 routines
  const T& gbmv(size_type kl, size_type ku, const value_type &alpha, const T &a,
                const T &x, const value_type &beta, const T&y);
  const T& gemv(const value_type &alpha, const T &a, const T &x,
                const value_type &beta, const T &y);
  const T& ger(const value_type &alpha, const T &x, const T &y, const T &a);
  const T& gerc(const value_type &alpha, const T &x, const T &y, const T &a);
};

}  // namespace linalg
}  // namespace thunder

#endif  // THUNDER_LINALG_LINALG_HPP_
