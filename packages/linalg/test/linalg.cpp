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

#include "gtest/gtest.h"
#include "thunder/linalg.hpp"
#include "thunder/linalg/blas.hpp"
#include "thunder/linalg/linalg.hpp"
#include "thunder/linalg/math.hpp"

TEST(LinalgTest, dummyTest) {
  int n = 100000;
  double x[100000];
  int incy = 1;
  double y[100000];
  int incx = 1;
  dswap_(&n, x, &incx, y, &incy);
  EXPECT_EQ(0, 0);
}

