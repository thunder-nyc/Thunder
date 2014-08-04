#ifndef THUNDER_STORAGE_HPP_
#define THUNDER_STORAGE_HPP_

#include <initializer_list>
#include <memory>

namespace thunder {

// Storage class declaration. If you find some member functions of this class
// looks like std::vector and std::valarray, it is indeed so. However, what we
// need is a std::valarray that supports customized allocator and iterators,
// therefore we have to implement this. The questions is, why doesn't the C++
// committee want std::valarray to be templated on an allocator?
template <typename T, typename Allocator = std::allocator<T> >
class Storage {
 public:
  // Typedefs from allocator
  typedef T value_type;
  typedef Allocator allocator_type;
  typedef typename Allocator::reference reference;
  typedef typename Allocator::const_reference const_reference;
  typedef typename Allocator::difference_type difference_type;
  typedef typename Allocator::size_type size_type;
  typedef std::allocator_traits<Allocator>::pointer pointer;
  typedef std::allocator_traits<Allocator>::const_pointer const_pointer;

  // Iterator definitions
  typedef pointer iterator;
  typedef const_pointer const_iterator;
  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

  // Default Constructor
  explicit Storage(const Allocator &alloc = Allocator());
  // Constructor with given size
  explicit Storage(size_type count, const Allocator &alloc = Allocator());
  // Constructor with given size and a default value
  explicit Storage(size_type count, const T &value,
                   const Allocator &alloc = Allocator());
  // Constructor with given iterator
  template <typename InputIt>
  vector(InputIt first, InputIt last, const Allocator &alloc = Allocator());
  // Copy constructors
  Storage(const Storage &other);
  Storage(const Storage &other, const Allocator &alloc);
  // Move constructors
  Storage(Storage &&other);
  Storage(Storage &&other, const Allocator &alloc);
  // Initializer list constructors
  Storage(std::initializer_list<T> init, const Allocator &alloc = Allocator());
  // Vector constructor: does not need to use the same allocator.
  template <typename VectorAllocator>
  Storage(std::vector<T, VectorAllocator> vect,
          const Allocator &alloc = Allocator());
  // Valarray constrctor
  Storage(std::valarray<T> vala, const Allocator &alloc = Allocator());

  // Destructor
  ~Storage();

  // Assignment operator
  Storage &operator=(const Storage &other);
  // Move assignment operator
  Storage &operator=(Storage && other);
  // Initializer list assignment operator
  Storage &operator=(std::initializer_list<T> ilist);
  // Vector assignment operator
  template <typename VectorAllocator>
  Storage &operator=(std::vector<T, VectorAllocator> vect);
  // Valarray assignment operator
  Storage &operator=(std::valarray<T> vala);

  // Asign with copy of count value
  void assign(size_type count, const T &value);
  // Assign with iterator
  template< class InputIt>
  void assign(InputIt first, InputIt last);
  // Assign to initializer list
  void assign(std::initializer_list<T> ilist);

  // Get allocator associated with the storage
  allocator_type get_allocator() const;

  // Get reference at pos with bound checking.
  reference at(size_type pos);
  // Get const reference at pos with bound checking.
  const_reference at(size_type pos) const;
  // Get reference at pos without bound checking
  reference operator[](size_type pos);
  // Get const reference at pos without bound checking
  const_reference operator[](size_type pos);

  // Get reference of first element
  reference front();
  // Get const reference of first element
  const_reference front() const;
  // Get reference of last element
  reference back();
  // Get const reference of last element
  const_reference back() const;

  // Get raw pointer to data
  pointer data();
  // Get const raw pointer to data
  const_pointer data() const;

  // Get iterator to data
  iterator begin();
  // Get const iterator to data
  const_iterator begin() const;
  const_iterator cbegin() const;
  // Get iterater pass the last element
  iterator end();
  // Get const iterator passing the last element
  const_iterator end() const;
  const_iterator cend() const;
  // Get reverse iterator pointing to last element
  reverse_iterator rbegin();
  // Get const reverse iterator pointing to the last element
  const_reverse_iterator rbegin() const;
  const_reverse_iterator crbegin() const;
  // Get reverse iterator passing the first element
  reverse_iterator rend();
  // Get const reverse iterator passing the first element
  const_reverse_iterator rend() const;
  const_reverse_iterator crend() const;

  // Check whether the storage is empty
  bool empty() const;
  // Check the size of the storage
  size_type size() const;
  // Return max_size. This returns std::numeric_limits<size_type>::max().
  size_type max_size() const;
  // Resize
  void resize(std::size_t count);
  // Resize with all elements using target value
  void resize(std::size_t count, const T &value);

  // Swap contents with another storage
  void swap(Storage &other);


 private:
  Allocator alloc_;
};

}  // namespace thunder

#endif  // THUNDER_STORAGE_HPP_
