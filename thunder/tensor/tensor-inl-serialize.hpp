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

#include "boost/mpl/integral_c.hpp"
#include "boost/mpl/integral_c_tag.hpp"
#include "boost/serialization/extended_type_info.hpp"
#include "boost/serialization/nvp.hpp"
#include "boost/serialization/singleton.hpp"
#include "boost/serialization/shared_ptr_helper.hpp"
#include "boost/serialization/split_free.hpp"
#include "boost/serialization/version.hpp"

namespace boost {
namespace serialization {

template< typename D, typename A >
struct version< ::std::shared_ptr< ::thunder::Storage< D, A > > > {
  typedef mpl::integral_c_tag tag;
  typedef mpl::int_< 1 > type;
  BOOST_STATIC_CONSTANT(int, value = type::value);
};

template< typename D, typename A >
struct tracking_level< ::thunder::Storage< D, A > > {
  typedef mpl::integral_c_tag tag;
  typedef mpl::int_< ::boost::serialization::track_never > type;
  BOOST_STATIC_CONSTANT(int, value = type::value);
};

template< typename C, typename D, typename A >
void save(
    C &ar, const ::std::shared_ptr< ::thunder::Storage< D, A > > &t,
    const unsigned int version) {
  const ::thunder::Storage< D, A >* t_ptr = t.get();
  ar << boost::serialization::make_nvp("px", t_ptr);
}

template< typename C, typename D, typename A >
void load(
    C &ar, ::std::shared_ptr< ::thunder::Storage< D, A > > &t,
    const unsigned int version) {
  ::thunder::Storage< D, A >* r;
  ar >> boost::serialization::make_nvp("px", r);
  boost::serialization::shared_ptr_helper< ::std::shared_ptr > & h =
      ar.template get_helper< shared_ptr_helper< ::std::shared_ptr > >();
  h.reset(t, r);
}

template< typename C, typename D, typename A >
void serialize(
    C &ar, ::std::shared_ptr< ::thunder::Storage< D, A > > &t,
    const unsigned int version) {
  boost::serialization::split_free(ar, t, version);
}

}  // namespace serialization
}  // namespace boost

namespace thunder {
namespace tensor {

template < typename S >
template < typename C >
void Tensor< S >::serialize(C &ar, const unsigned int version) {
  ar & size_;
  ar & stride_;
  ar & storage_;
  ar & offset_;
}

}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_INL_SERIALIZE_HPP_
