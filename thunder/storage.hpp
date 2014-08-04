#ifndef THUNDER_STORAGE_HPP_
#define THUNDER_STORAGE_HPP_

#include <memory>

namespace thunder {

template <typename T, typename Allocator = std::allocator<T> >
class Storage {
 public:
  typedef typename Allocator::value_type value_type;
  typedef typename Allocator::reference reference;
  typedef typename Allocator::const_reference const_reference;
  typedef typename Allocator::difference_type difference_type;
  typedef typename Allocator::size_type size_type;

  // Default Constructor
  explicit Storage(const Allocator &alloc = Allocator());

  // Constructor with given size
  explicit Storage(size_type count, const Allocator &alloc = Allocator());

  // Constructor with given size and a default value
  explicit Storage(size_type count, const T &value,
                   const Allocator &alloc = Allocator());

  // Copy constructors
  Storage(const Storage &other);
  Storage(const Storage &other, const Allocator &alloc);

  // Move constructors
  Storage(Storage &&other);
  Storage(Storage &&other, const Allocator &alloc);

  // Initializer list constructors
  Storage(std::initializer_list<T> init, const Allocator &alloc = Allocator())

  virtual ~Storage() {};

 private:
  Allocator alloc_;
}

}  // namespace thunder

#endif  // THUNDER_STORAGE_HPP_
