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

#include "thunder/serializer/text_protocol.hpp"

#include <sstream>

#include "gtest/gtest.h"
#include "thunder/serializer/text_protocol-inl.hpp"

namespace thunder {
namespace serializer {

template < typename T >
void charTest() {
  TextProtocol< ::std::stringstream> t;

  T single_saved = 'A';
  t.save(&single_saved, single_saved);
  T single_loaded = '0';
  t.load(&single_saved, &single_loaded);
  EXPECT_EQ(single_saved, single_loaded);

  T array_saved[5];
  T array_loaded[5];
  for (int i = 0; i < 5; ++i) {
    array_saved[i] = 'a' + i;
    array_loaded[i] = '0';
    t.save(&single_saved, array_saved[i]);
  }
  for (int i = 0; i < 5; ++i) {
    t.load(&single_saved, &array_loaded[i]);
    EXPECT_EQ(array_saved[i], array_loaded[i]);
  }
}

TEST(TextProtocolTest, charTest) {
  charTest< char >();
  charTest< signed char >();
  charTest< unsigned char >();
  charTest< wchar_t >();
  charTest< char16_t >();
  charTest< char32_t >();
}

template < typename T >
void intTest() {
  TextProtocol< ::std::stringstream> t;

  T single_saved = 9;
  t.save(&single_saved, single_saved);
  T single_loaded = 0;
  t.load(&single_saved, &single_loaded);
  EXPECT_EQ(single_saved, single_loaded);

  T array_saved[7];
  T array_loaded[7];
  for (int i = 0; i < 7; ++i) {
    array_saved[i] = 18 + i;
    array_loaded[i] = 0;
    t.save(&single_saved, array_saved[i]);
  }
  for (int i = 0; i < 7; ++i) {
    t.load(&single_saved, &array_loaded[i]);
    EXPECT_EQ(array_saved[i], array_loaded[i]);
  }
}

TEST(TextProtocolTest, intTest) {
  intTest< short >();
  intTest< unsigned short >();
  intTest< int >();
  intTest< unsigned int >();
  intTest< long >();
  intTest< unsigned long >();
  intTest< long long >();
  intTest< unsigned long long >();
}

template < typename T >
void floatTest() {
  TextProtocol< ::std::stringstream> t;

  T single_saved = 9.1154684568;
  t.save(&single_saved, single_saved);
  T single_loaded = 0.0;
  t.load(&single_saved, &single_loaded);
  EXPECT_EQ(single_saved, single_loaded);

  T array_saved[9];
  T array_loaded[9];
  for (int i = 0; i < 9; ++i) {
    array_saved[i] = (-4.5443279658 + i) / 1.798;
    array_loaded[i] = 0.0;
    t.save(&single_saved, array_saved[i]);
  }
  for (int i = 0; i < 7; ++i) {
    t.load(&single_saved, &array_loaded[i]);
    EXPECT_EQ(array_saved[i], array_loaded[i]);
  }
}

TEST(TextProtocolTest, floatTest) {
  floatTest< float >();
  floatTest< double >();
  floatTest< long double >();
}

// For libstdc++ we can only capture ::std::exception here due to ABI
// imcompatibility between C++98 and C++11.
// Reference: https://gcc.gnu.org/PR66145
template < typename T >
void exceptionTest() {
  typedef TextProtocol< ::std::stringstream > P;
  T saved = 9;
  T loaded = 0;

  P p1(P::in);
  EXPECT_THROW({
      p1.save(&saved, saved);
    }, ::std::exception);

  P p2(P::out);
  p2.save(&saved, saved);
  EXPECT_THROW({
      p2.load(&saved, &loaded);
    }, ::std::exception);
}

TEST(TextProtocolTest, exceptionTest) {
  exceptionTest< int >();
}

}  // namespace serializer
}  // namespace thunder
