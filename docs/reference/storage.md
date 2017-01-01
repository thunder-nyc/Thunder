# Thunder-Storage

The Thunder-Storage package contains only one class -- `Storage`, which handles actual data in memory as an dynamically allocated array.

## template < typename D = double, typename A = ::std::allocator < D > > class Storage();

The storage class. The template parameter `D` specifies a data type, and the `A` specifies an allocator. The allocator can be used to distinguish between data allocated on different devices, such as those on CPUs and on GPUs. The allocator class should follow similar protocol as the [standard C++11 allocator](http://en.cppreference.com/w/cpp/memory/allocator).

### Instantiated Storage classes

The following storage classes are instantiated and ready for use in the `::thunder` namespace. The compiler will find them in the `libthunder_storage.so` dynamic library instead of compiling from scratch.

| Name (in `thunder` namespace)   | Definition (in `thunder::storage` nemaspace)                                     |
|---------------------------------|----------------------------------------------------------------------------------|
| DoubleStorage                   | Storage< double, ::std::alocator< double > >                                     |
| FloatStorage                    | Storage< float, ::std::alocator< float > >                                       |
| SizeStorage                     | Storage< ::std::size_t, ::std::alocator< ::std::size_t > >                       |
| DoubleComplexStorage            | Storage< ::std::complex< double >, ::std::alocator< ::std::complex< double > > > |
| FloatComplexStorage             | Storage< ::std::complex< float >, ::std::alocator< ::std::complex< float > > >   |

### Symbols defined in Storage class

The following symbols are defined in the storage class.

| Symbol               | Description                                                                                       |
|----------------------|---------------------------------------------------------------------------------------------------|
| allocator_type       | The allocator type. A `typedef` of `A`.                                                           |
| value_type           | The value type. A `typedef` of `A::value_type`.                                                   |
| reference            | The reference type. A `typedef` of `A::reference`.                                                |
| const_reference      | The const reference type. A `typedef` of `A::const_reference`.                                    |
| difference_type      | The difference type. A `typedef` of `A::difference`.                                              |
| size_type            | The size type. A `typedef` of `A::size_type`.                                                     |
| pointer              | The pointer type. A `typedef` of `::std::allocator_traits< A >::pointer`.                         |
| const_pointer        | The const pointer type. A `typedef` of `::std::allocator_traits< A >::const_pointer`.             |
| shared_pointer       | The shared_pointer type. A `typedef` of `::std::shared_ptr< value_type >`.                        |
| iterator             | The value iterator type. A `typedef` of `pointer`.                                                |
| const_iterator       | The const value iterator type. A `typedef` of `const_pointer`.                                    |

### Constructors

The `Storage` class has the following constructors.

#### explicit Storage(A alloc = A());

Default constructor with optional allocator.

#### explicit Storage(size_type count, A alloc = A());

Constructor with given data size.

#### Storage(size_type count, const_reference value, A alloc = A());

Constructor with given data size and a default value.

#### Storage(shared_pointer shared, size_type count, A alloc = A());

Constructor with given shared pointer and data size.

#### Storage(pointer data, size_type count, A alloc = A());

Constructor with given pointer and data size. The data will be managed by the storage object, therefore you should not access the pointer any more.

#### Storage(const Storage &other);

Copy-constructor.

#### Storage(Storage &&other);

Move-constructor.

#### Storage(::std::initializer_list< D > init, A alloc = A());

Constructor from an initializer list. The storage will contain the same values as in the initializer list.

#### ~Storage()

Destructor.

### Operators

The `Storage` class has the following operators.

#### Storage &operator=(Storage other);

Assignment operator using copy and swap idiom.

#### reference operator\[](size_type pos);
#### const_reference operator\[](size_type pos) const;

Get reference at `pos` without bound checking.

### Property Queries

The following functions can be used to query properties of the `Storage` class.

#### pointer data() const;

Get the raw pointer to underlying data.

#### shared_pointer shared() const;

Get the shared pointer to the underlying data.

#### size_type size() const;

Get the size of the storage.

#### A allocator() const;

Get the allocator.

### Iterators

The `Storage` class contans the following iterator functions.

#### iterator begin();

Get the iterator to data.

#### const_iterator begin() const;

Get the const iterator to data.

#### iterator end();

Get the iterator passing the last element.

#### iterator end();

Get the const iterator passing the last element.

### Other member functions

The `Storage` class contains the following member functions that may be useful.

#### template< typename S > void copy(const S &other);

Copy from another storage using `static_cast`.

#### void resize(size_type count);

Resize the storage. Data content will be lost.

#### void resize(size_type count, const_reference value);

Resize the storage and initialize them with `value`.

#### template < typename S > S view();

View the storage as a different storage by applying `reinterpret_cast` to the underlying data pointer. The new storage still shares the underlying space. This is used in Thunder-Tensor to view a complex tensor as a non-complex one.
