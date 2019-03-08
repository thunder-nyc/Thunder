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

#ifndef THUNDER_SERIALIZER_SERIALIZER_INL_HPP_
#define THUNDER_SERIALIZER_SERIALIZER_INL_HPP_

#include "thunder/serializer/serializer.hpp"

#include <memory>

namespace thunder {
namespace serializer {

template < typename P >
template < typename... G >
Serializer< P >::Serializer(G ...g) :
    protocol_(g...), saved_count_(0) {}

template < typename P >
const typename Serializer< P >::protocol_type& Serializer< P >::protocol()
    const {
  return protocol_;
}

template < typename P >
template < typename T >
void Serializer< P >::save(const T &t) {
  protocol_.save(this, t);
}

template < typename P >
template < typename T >
void Serializer< P >::load(T *t) {
  protocol_.load(this, t);
}

template < typename P >
template < typename T >
void Serializer< P >::save(T* const &t) {
  if (saved_pointers_.find(static_cast< void* >(t)) == saved_pointers_.end()) {
    // Special key 0 mark new serializer record to support appending
    if (saved_count_ == 0) {
      protocol_.save(this, saved_count_);
    }
    unsigned int key = ++saved_count_;
    saved_pointers_[static_cast< void* >(t)] = key;
    protocol_.save(this, key);
    protocol_.save(this, *t);
  } else {
    protocol_.save(this, saved_pointers_[static_cast< void * >(t)]);
  }
}

template < typename P >
template < typename T >
void Serializer< P >::load(T* *t) {
  unsigned int key;
  protocol_.load(this, &key);

  // When key is 0, a new serializer record. Clean up the tables!
  if (key == 0) {
    loaded_pointers_.clear();
    loaded_shared_.clear();
    protocol_.load(this, &key);
  }

  // Check the tables and load the data
  if (loaded_pointers_.find(key) == loaded_pointers_.end() &&
      loaded_shared_.find(key) == loaded_shared_.end()) {
    *t = new T();
    protocol_.load(this, *t);
    loaded_pointers_[key] = static_cast< void* >(*t);
  } else if (loaded_pointers_.find(key) == loaded_pointers_.end()) {
    *t = static_cast< T* >(static_cast< ::std::shared_ptr< T >* >(
        loaded_shared_[key])->get());
    loaded_pointers_[key] = static_cast< void* >(*t);
  } else {
    *t = static_cast< T* >(loaded_pointers_[key]);
  }
}

template < typename P >
template < typename T >
void Serializer< P >::save(const ::std::shared_ptr< T > &t) {
  if (saved_pointers_.find(static_cast< void* >(t.get())) ==
      saved_pointers_.end()) {
    // Special key 0 appended if this archive is a new record
    if (saved_count_ == 0) {
      protocol_.save(this, saved_count_);
    }
    unsigned int key = ++saved_count_;
    saved_pointers_[static_cast< void* >(t.get())] = key;
    protocol_.save(this, key);
    save(*t);
  } else {
    protocol_.save(this, saved_pointers_[static_cast< void * >(t.get())]);
  }
}

template < typename P >
template < typename T >
void Serializer< P >::load(::std::shared_ptr< T > *t) {
  unsigned int key;
  protocol_.load(this, &key);

  // If key is 0, new serialization record. Clean up the tables!
  if (key == 0) {
    loaded_pointers_.clear();
    loaded_shared_.clear();
    protocol_.load(this, &key);
  }

  // Check the tables and load the data
  if (loaded_shared_.find(key) == loaded_shared_.end() &&
      loaded_pointers_.find(key) == loaded_pointers_.end()) {
    *t = ::std::make_shared< T >();
    protocol_.load(this, t->get());
    loaded_shared_[key] = static_cast< void* >(t);
  } else if (loaded_shared_.find(key) == loaded_shared_.end()) {
    *t = ::std::shared_ptr< T >(static_cast< T* >(loaded_pointers_[key]));
    loaded_shared_[key] = static_cast< void* >(t);
  } else {
    *t = *(static_cast< ::std::shared_ptr< T >* >(loaded_shared_[key]));
  }
}

}  // namespace serializer
}  // namespace thunder

#endif  // THUNDER_SERIALIZER_SERIALIZER_INL_HPP_
