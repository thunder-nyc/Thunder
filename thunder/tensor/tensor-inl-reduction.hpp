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

#ifndef THUNDER_TENSOR_TENSOR_INL_REDUCTION_HPP_
#define THUNDER_TENSOR_TENSOR_INL_REDUCTION_HPP_

#include "thunder/tensor/tensor.hpp"
#include "thunder/tensor/tensor-inl.hpp"

namespace thunder {
namespace tensor {

virtual value_type max(Tensor< size_type > *pos = nullptr) const;
virtual value_type min(Tensor< size_type > *pos = nullptr) const;
virtual value_type sum() const;
virtual value_type prod() const;
virtual value_type mean() const;
virtual value_type var() const;
virtual value_type std() const;

static value_type max(const Tensor &x, Tensor< size_type > *pos = nullptr);
static value_type min(const Tensor &x, Tensor< size_type > *pos = nullptr);
static value_type sum(const Tensor &x);
static value_type prod(const Tensor &x);
static value_type mean(const Tensor &x);
static value_type var(const Tensor &x);
static value_type std(const Tensor &x);

virtual value_type max(dim_type d, Tensor< size_type > *pos = nullptr) const;
virtual value_type min(dim_type d, Tensor< size_type > *pos = nullptr) const;
virtual value_type sum(dim_type d) const;
virtual value_type prod(dim_type d) const;
virtual value_type mean(dim_type d) const;
virtual value_type var(dim_type d) const;
virtual value_type std(dim_type d) const;

static value_type max(const Tensor &x, dim_type d,
                      Tensor< size_type > *pos = nullptr);
static value_type min(const Tensor &x, dim_type d,
                      Tensor< size_type > *pos = nullptr);
static value_type sum(const Tensor &x, dim_type d);
static value_type prod(const Tensor &x, dim_type d);
static value_type mean(const Tensor &x, dim_type d);
static value_type var(const Tensor &x, dim_type d);
static value_type std(const Tensor &x, dim_type d);

}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_INL_REDUCTION_HPP_
