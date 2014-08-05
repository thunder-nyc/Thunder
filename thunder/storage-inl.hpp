/*
 * \copyright Copyright 2014 Xiang Zhang. All Rights Reserved.
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

#ifndef THUNDER_STORAGE_INL_HPP
#define THUNDER_STORAGE_INL_HPP

namespace thunder {

template <typename T, typename Allocator = std::allocator<T> >
Storage <T, Allocator>::Storage(const Allocator &alloc = Allocator())
    : alloc_(alloc), data_(nullptr) {}

template <typename T, typename Allocator = std::allocator<T> >
Storage<T, Allocator>::Storage(size_type count,
                               const Allocator &alloc = Allocator())
    : alloc_(alloc) {
  data_ = alloc_.allocate(count);
}

}  // namespace thunder

#endif  // THUNDER_STORAGE_INL_HPP
