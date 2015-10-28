/*
 * \copyright Copyright 2014-2015 Xiang Zhang All Rights Reserved.
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

#ifndef THUNDER_LINALG_COMPLEX_HPP_
#define THUNDER_LINALG_COMPLEX_HPP_

#include "thunder/linalg/linalg.hpp"
#include "thunder/storage.hpp"
#include "thunder/tensor.hpp"

namespace thunder {
namespace linalg {
namespace math {

template < typename D, typename A, typename H >
const Tensor< Storage< ::std::complex< D >, A > >& rotm(
    Linalg< Tensor< Storage < ::std::complex< D >, A > >, H > *l,
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y,
    const Tensor< Storage< ::std::complex< D >, A > > &p);

}  // namespace math
}  // namespace linalg
}  // namespace thunder

#endif  // THUNDER_LINALG_COMPLEX_HPP_
