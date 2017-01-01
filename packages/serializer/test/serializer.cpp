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

#include "thunder/serializer/serializer.hpp"

#include <exception>
#include <ios>
#include <memory>
#include <sstream>
#include <string>

#include "gtest/gtest.h"
#include "thunder/serializer.hpp"
#include "thunder/serializer/serializer-inl.hpp"
#include "thunder/serializer/text_protocol.hpp"
#include "thunder/serializer/text_protocol-inl.hpp"

namespace thunder {
namespace serializer {

template < typename S, typename T >
void charTest() {
  S s;

  T single_saved = 'A';
  s.save(single_saved);
  T single_loaded = '0';
  s.load(&single_loaded);
  EXPECT_EQ(single_saved, single_loaded);

  T array_saved[5];
  T array_loaded[5];
  for (int i = 0; i < 5; ++i) {
    array_saved[i] = 'a' + i;
    array_loaded[i] = '0';
    s.save(array_saved[i]);
  }
  for (int i = 0; i < 5; ++i) {
    s.load(&array_loaded[i]);
    EXPECT_EQ(array_saved[i], array_loaded[i]);
  }
}

TEST(SerializerTest, charTest) {
  charTest< StringTextSerializer, char >();
  charTest< StringTextSerializer, signed char >();
  charTest< StringTextSerializer, unsigned char >();
  charTest< StringTextSerializer, wchar_t >();
  charTest< StringTextSerializer, char16_t >();
  charTest< StringTextSerializer, char32_t >();
}

template < typename S, typename T >
void intTest() {
  S s;

  T single_saved = 9;
  s.save(single_saved);
  T single_loaded = 0;
  s.load(&single_loaded);
  EXPECT_EQ(single_saved, single_loaded);

  T array_saved[7];
  T array_loaded[7];
  for (int i = 0; i < 7; ++i) {
    array_saved[i] = -3 + i;
    array_loaded[i] = 0;
    s.save(array_saved[i]);
  }
  for (int i = 0; i < 7; ++i) {
    s.load(&array_loaded[i]);
    EXPECT_EQ(array_saved[i], array_loaded[i]);
  }
}

TEST(SerializerTest, intTest) {
  intTest< StringTextSerializer, short >();
  intTest< StringTextSerializer, unsigned short >();
  intTest< StringTextSerializer, int >();
  intTest< StringTextSerializer, unsigned int >();
  intTest< StringTextSerializer, long >();
  intTest< StringTextSerializer, unsigned long >();
  intTest< StringTextSerializer, long long >();
  intTest< StringTextSerializer, unsigned long long >();
}

template < typename S, typename T >
void floatTest() {
  S s;

  T single_saved = 9.1154684568;
  s.save(single_saved);
  T single_loaded = 0.0;
  s.load(&single_loaded);
  EXPECT_EQ(single_saved, single_loaded);

  T array_saved[7];
  T array_loaded[7];
  for (int i = 0; i < 7; ++i) {
    array_saved[i] = (-4.5443279658 + i) / 1.798;
    array_loaded[i] = 0;
    s.save(array_saved[i]);
  }
  for (int i = 0; i < 7; ++i) {
    s.load(&array_loaded[i]);
    EXPECT_EQ(array_saved[i], array_loaded[i]);
  }
}

TEST(SerializerTest, floatTest) {
  floatTest< StringTextSerializer, float >();
  floatTest< StringTextSerializer, double >();
  floatTest< StringTextSerializer, long double >();
}

template < typename S, typename T >
void pointerTest() {
  S s;

  T* pointer_saved1 = new T(3);
  T* pointer_saved2 = new T(9);
  s.save(pointer_saved1);
  s.save(pointer_saved1);
  s.save(pointer_saved2);

  T* pointer_loaded1 = nullptr;
  T* pointer_loaded2 = nullptr;
  T* pointer_loaded3 = nullptr;
  s.load(&pointer_loaded1);
  s.load(&pointer_loaded2);
  s.load(&pointer_loaded3);
  EXPECT_EQ(*pointer_saved1, *pointer_loaded1);
  EXPECT_EQ(*pointer_saved1, *pointer_loaded2);
  EXPECT_EQ(*pointer_saved2, *pointer_loaded3);
  EXPECT_NE(pointer_saved1, pointer_loaded1);
  EXPECT_NE(pointer_saved1, pointer_loaded2);
  EXPECT_NE(pointer_saved2, pointer_loaded3);
  EXPECT_EQ(pointer_loaded1, pointer_loaded2);
  EXPECT_NE(pointer_loaded1, pointer_loaded3);

  // Take the string and put it to another stream
  ::std::string str = s.protocol().stream().str();
  S s1(str, ::std::ios_base::in | ::std::ios_base::out | ::std::ios_base::ate);

  ::std::shared_ptr< T > pointer_saved3 = ::std::make_shared< T >(6);
  T* pointer_saved4 = new T(8);
  s1.save(pointer_saved3);
  s1.save(pointer_saved3);
  s1.save(pointer_saved4);

  T* pointer_loaded4 = nullptr;
  ::std::shared_ptr< T > pointer_loaded5;
  T* pointer_loaded6 = nullptr;
  s1.load(&pointer_loaded4);
  s1.load(&pointer_loaded5);
  s1.load(&pointer_loaded6);
  EXPECT_EQ(*pointer_saved1, *pointer_loaded4);
  EXPECT_EQ(*pointer_saved1, *pointer_loaded5);
  EXPECT_EQ(*pointer_saved2, *pointer_loaded6);
  EXPECT_NE(pointer_saved1, pointer_loaded4);
  EXPECT_NE(pointer_saved1, pointer_loaded5.get());
  EXPECT_NE(pointer_saved2, pointer_loaded6);
  EXPECT_EQ(pointer_loaded4, pointer_loaded5.get());
  EXPECT_NE(pointer_loaded4, pointer_loaded6);
  EXPECT_NE(pointer_loaded1, pointer_loaded4);
  EXPECT_NE(pointer_loaded6, pointer_loaded3);

  ::std::shared_ptr< T > pointer_loaded7;
  T* pointer_loaded8 = nullptr;
  T* pointer_loaded9 = nullptr;
  s1.load(&pointer_loaded7);
  s1.load(&pointer_loaded8);
  s1.load(&pointer_loaded9);
  EXPECT_EQ(*pointer_saved3, *pointer_loaded7);
  EXPECT_EQ(*pointer_saved3, *pointer_loaded8);
  EXPECT_EQ(*pointer_saved4, *pointer_loaded9);
  EXPECT_NE(pointer_saved3.get(), pointer_loaded7.get());
  EXPECT_NE(pointer_saved3.get(), pointer_loaded8);
  EXPECT_NE(pointer_saved4, pointer_loaded9);
  EXPECT_EQ(pointer_loaded7.get(), pointer_loaded8);
  EXPECT_NE(pointer_loaded7.get(), pointer_loaded9);

  delete pointer_saved1;
  delete pointer_saved2;
  delete pointer_loaded1;
  delete pointer_loaded3;
  delete pointer_saved4;
  delete pointer_loaded6;
  delete pointer_loaded9;
}

TEST(SerializerTest, pointerTest) {
  pointerTest< StringTextSerializer, int >();
}

// For libstdc++ we can only capture ::std::exception here due to ABI
// imcompatibility between C++98 and C++11.
// Reference: https://gcc.gnu.org/PR66145
template < typename S, typename T >
void exceptionTest() {
  T saved = 9;
  T loaded = 0;

  S s1(S::in);
  EXPECT_THROW({
      s1.save(saved);
    }, ::std::exception);

  S s2(S::out);
  s2.save(saved);
  EXPECT_THROW({
      s2.load(&loaded);
    }, ::std::exception);
}

TEST(SerializerTest, exceptionTest) {
  exceptionTest< StringTextSerializer, int >();
}

}  // namespace serializer
}  // namespace thunder
