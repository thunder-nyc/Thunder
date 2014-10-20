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

#include "thunder/tensor/index_iterator.hpp"

#include <stddef.h>

#include "gtest/gtest.h"
#include "thunder/storage.hpp"

namespace thunder {
namespace tensor {
namespace {

template < typename S >
void indexIteratorTest() {
  S size({2, 4, 5});
  IndexIterator< S > begin(size);
  int i = 0;
  for (typename S::const_reference v : *begin) {
    EXPECT_EQ(0, v);
    EXPECT_EQ(begin->data()[i], v);
    ++i;
  }
  EXPECT_EQ(3, i);
  IndexIterator< S > end(size, 40);
  EXPECT_EQ(2, (*end)[0]);
  EXPECT_EQ(0, (*end)[1]);
  EXPECT_EQ(0, (*end)[2]);
  EXPECT_TRUE(begin != end);
  EXPECT_FALSE(begin == end);
  IndexIterator< S > current(size, *end);
  EXPECT_EQ(2, (*current)[0]);
  EXPECT_EQ(0, (*current)[1]);
  EXPECT_EQ(0, (*current)[2]);
  EXPECT_FALSE(end != current);
  EXPECT_TRUE(end == current);

  IndexIterator< S > static_begin = IndexIterator< S >::begin(size);
  IndexIterator< S > static_end = IndexIterator< S >::end(size);
  EXPECT_TRUE(static_begin == begin);
  EXPECT_TRUE(static_end == end);

  for (i = 0; begin != end; ++begin, ++i) {
    EXPECT_EQ(i, (*begin)[0] * 20 + (*begin)[1] * 5 + (*begin)[2]);
  }
  EXPECT_EQ(40, i);
}

TEST(IndexIteratorTest, indexIteratorTest) {
  indexIteratorTest< Storage< ::std::size_t > >();
}

}  // namespace
}  // namespace tensor
}  // namespace thunder
