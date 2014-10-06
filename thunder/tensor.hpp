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

#ifndef THUNDER_TENSOR_HPP
#define THUNDER_TENSOR_HPP

#include "thunder/tensor/tensor.hpp"

#include <complex>

#include "thunder/storage.hpp"

namespace thunder {

template < typename S = DoubleStorage >
using Tensor = tensor::Tensor< S >;
typedef Tensor< DoubleStorage > DoubleTensor;
typedef Tensor< FloatStorage > FloatTensor;

template < typename D = double,
           typename A = ::std::allocator< ::std::complex< D > > >
using ComplexTensor = tensor::Tensor< ComplexStorage< D, A > >;
typedef ComplexTensor< double, ::std::allocator< ::std::complex< double > > >
DoubleComplexTensor;
typedef ComplexTensor< float, ::std::allocator< ::std::complex< float > > >
FloatComplexTensor;

}  // namespace thunder

#endif  // THUNDER_TENSOR_HPP
