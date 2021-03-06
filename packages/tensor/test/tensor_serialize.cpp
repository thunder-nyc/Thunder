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

#include "thunder/tensor.hpp"

#include <memory>
#include <typeinfo>

#include "gtest/gtest.h"
#include "thunder/serializer.hpp"
#include "thunder/storage.hpp"

namespace thunder {
namespace {

template< typename T >
void serializeTest() {
  T t1({10, 20, 7}, {161, 8, 1});
  int t1_val = -800;
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t1_val++) /
        static_cast< typename T::value_type >(300);
  }

  StringBinarySerializer s1;
  s1.save(t1);
  T t2;
  s1.load(&t2);
  EXPECT_EQ(t1.dimension(), t2.dimension());
  EXPECT_EQ(t1.offset(), t2.offset());
  for (int i = 0; i < t1.dimension(); ++i) {
    EXPECT_EQ(t1.size(i), t2.size(i));
    EXPECT_EQ(t1.stride(i), t2.stride(i));
  }
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(), t2_begin = t2.reference_begin();
       begin != end; ++begin, ++t2_begin) {
    EXPECT_EQ(*begin, *t2_begin);
  }

  StringTextSerializer s2;
  s2.save(t1);
  T t3;
  s2.load(&t3);
  EXPECT_EQ(t1.dimension(), t3.dimension());
  EXPECT_EQ(t1.offset(), t3.offset());
  for (int i = 0; i < t1.dimension(); ++i) {
    EXPECT_EQ(t1.size(i), t3.size(i));
    EXPECT_EQ(t1.stride(i), t3.stride(i));
  }
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(), t3_begin = t3.reference_begin();
       begin != end; ++begin, ++t3_begin) {
    EXPECT_EQ(*begin, *t3_begin);
  }
}

TEST(TensorTest, serializeTest) {
  serializeTest< DoubleTensor >();
  serializeTest< FloatTensor >();
  serializeTest< DoubleComplexTensor >();
  serializeTest< FloatComplexTensor >();
}

}  // namespace
}  // namespace thunder
