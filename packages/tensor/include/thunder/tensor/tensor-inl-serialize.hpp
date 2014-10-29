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

#ifndef THUNDER_TENSOR_TENSOR_INL_SERIALIZE_HPP_
#define THUNDER_TENSOR_TENSOR_INL_SERIALIZE_HPP_

#include "thunder/tensor/tensor.hpp"
#include "thunder/tensor/tensor-inl.hpp"

#include <memory>

#include "thunder/serializer.hpp"

namespace thunder {
namespace serializer {

template < typename C, typename S >
void save(C *s, const tensor::Tensor< S > &t) {
  s->save(t.size());
  s->save(t.stride());
  s->save(t.storage());
  s->save(t.offset());
}

template < typename C, typename S >
void load(C *s, tensor::Tensor< S > *t) {
  typedef tensor::Tensor< S > T;

  typename T::size_storage size;
  s->load(&size);
  typename T::stride_storage stride;
  s->load(&stride);
  typename T::storage_pointer storage;
  s->load(&storage);
  typename T::size_type offset;
  s->load(&offset);

  t->set(size, stride, storage, offset);
}

}  // namespace serializer
}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_INL_SERIALIZE_HPP_
