/*
 * \copyright Copyright 2014 - 2015 Xiang Zhang All Rights Reserved.
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

#include <random>

#include "gtest/gtest.h"
#include "thunder/storage.hpp"

namespace thunder {
namespace {

template< typename T >
void sortTest() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);

  T t1(10, 20, 7);
  Tensor< typename T::size_storage > t1_index;

  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(dist(gen));
  }

  T t1_sorted = T::sort(t1, 1);
  for (int i = 0; i < t1.size(0); ++i) {
    for (int j = 0; j < t1.size(1) - 1; ++j) {
      for (int k = 0; k < t1.size(2); ++k) {
        EXPECT_LE(t1_sorted(i, j, k), t1_sorted(i, j + 1, k));
      }
    }
  }

  T t1_revert = T::sort(t1, 1, true);
  for (int i = 0; i < t1.size(0); ++i) {
    for (int j = 0; j < t1.size(1) - 1; ++j) {
      for (int k = 0; k < t1.size(2); ++k) {
        EXPECT_GE(t1_revert(i, j, k), t1_revert(i, j + 1, k));
      }
    }
  }

  t1_sorted = T::sort(t1, 1, &t1_index);
  for (int i = 0; i < t1.dimension(); ++i) {
    EXPECT_EQ(t1_index.size(i), t1.size(i));
  }
  for (int i = 0; i < t1.size(0); ++i) {
    for (int j = 0; j < t1.size(1); ++j) {
      for (int k = 0; k < t1.size(2); ++k) {
        EXPECT_FLOAT_EQ(t1_sorted(i, j, k), t1(i, t1_index(i, j, k), k));
      }
    }
  }
  for (int i = 0; i < t1.size(0); ++i) {
    for (int j = 0; j < t1.size(1) - 1; ++j) {
      for (int k = 0; k < t1.size(2); ++k) {
        EXPECT_LE(t1_sorted(i, j, k), t1_sorted(i, j + 1, k));
      }
    }
  }

  t1_revert = T::sort(t1, 1, &t1_index, true);
  for (int i = 0; i < t1.dimension(); ++i) {
    EXPECT_EQ(t1_index.size(i), t1.size(i));
  }
  for (int i = 0; i < t1.size(0); ++i) {
    for (int j = 0; j < t1.size(1); ++j) {
      for (int k = 0; k < t1.size(2); ++k) {
        EXPECT_FLOAT_EQ(t1_revert(i, j, k), t1(i, t1_index(i, j, k), k));
      }
    }
  }
  for (int i = 0; i < t1.size(0); ++i) {
    for (int j = 0; j < t1.size(1) - 1; ++j) {
      for (int k = 0; k < t1.size(2); ++k) {
        EXPECT_GE(t1_revert(i, j, k), t1_revert(i, j + 1, k));
      }
    }
  }

  T t2({3, 10, 20, 7}, {5200, 408, 19, 2});
  Tensor< typename T::size_storage > t2_index;

  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(dist(gen));
  }

  T t2_sorted = T::sort(t2, 2);
  for (int l = 0; l < t2.size(0); ++l) {
    for (int i = 0; i < t2.size(1); ++i) {
      for (int j = 0; j < t2.size(2) - 1; ++j) {
        for (int k = 0; k < t2.size(3); ++k) {
          EXPECT_LE(t2_sorted(l, i, j, k), t2_sorted(l, i, j + 1, k));
        }
      }
    }
  }

  T t2_revert = T::sort(t2, 2, true);
  for (int l = 0; l < t2.size(0); ++l) {
    for (int i = 0; i < t2.size(1); ++i) {
      for (int j = 0; j < t2.size(2) - 1; ++j) {
        for (int k = 0; k < t2.size(3); ++k) {
          EXPECT_GE(t2_revert(l, i, j, k), t2_revert(l, i, j + 1, k));
        }
      }
    }
  }

  t2_sorted = T::sort(t2, 2, &t2_index);
  for (int i = 0; i < t2.dimension(); ++i) {
    EXPECT_EQ(t2_index.size(i), t2.size(i));
  }
  for (int l = 0; l < t2.size(0); ++l) {
    for (int i = 0; i < t2.size(1); ++i) {
      for (int j = 0; j < t2.size(2); ++j) {
        for (int k = 0; k < t2.size(3); ++k) {
          EXPECT_FLOAT_EQ(
              t2_sorted(l, i, j, k), t2(l, i, t2_index(l, i, j, k), k));
        }
      }
    }
  }
  for (int l = 0; l < t2.size(0); ++l) {
    for (int i = 0; i < t2.size(1); ++i) {
      for (int j = 0; j < t2.size(2) - 1; ++j) {
        for (int k = 0; k < t2.size(3); ++k) {
          EXPECT_LE(t2_sorted(l, i, j, k), t2_sorted(l, i, j + 1, k));
        }
      }
    }
  }

  t2_revert = T::sort(t2, 2, &t2_index, true);
  for (int i = 0; i < t2.dimension(); ++i) {
    EXPECT_EQ(t2_index.size(i), t2.size(i));
  }
  for (int l = 0; l < t2.size(0); ++l) {
    for (int i = 0; i < t2.size(1); ++i) {
      for (int j = 0; j < t2.size(2); ++j) {
        for (int k = 0; k < t2.size(3); ++k) {
          EXPECT_FLOAT_EQ(
              t2_revert(l, i, j, k), t2(l, i, t2_index(l, i, j, k), k));
        }
      }
    }
  }
  for (int l = 0; l < t2.size(0); ++l) {
    for (int i = 0; i < t2.size(1); ++i) {
      for (int j = 0; j < t2.size(2) - 1; ++j) {
        for (int k = 0; k < t2.size(3); ++k) {
          EXPECT_GE(t2_revert(l, i, j, k), t2_revert(l, i, j + 1, k));
        }
      }
    }
  }
}

TEST(TensorTest, sortTest) {
  sortTest< DoubleTensor >();
  sortTest< FloatTensor >();
}

}  // namespace
}  // namespace thunder
