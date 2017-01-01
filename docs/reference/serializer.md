# Thunder-Serializer

The Thunder-Serializer package is a package used by many other Thunder packages for serialization data to strings or files. The goal of Thunder-Serializer includes

* Serialization of all fundamental types in C++11.
* Preserving data sharing relationships by only saving pointer memory once.
* Dynamic serialization of pointers that bind polymorphic types.
* Providing a non-intrusive interface for user-defined classes.

The current Thunder-Serializer implementation is only experimental, in that polymorphic serialization is not yet implemented. That said, further development of Thunder-Serializer will ensure backward compatibility.

## Using Thunder-Serializer

Thunder-Serializer package contains the the following four classes for serialization. They are available in the `thunder` namespace if you include `thunder/serializer.hpp`.

* `StringBinarySerializer`: a serializer that serializes data into a string stream in binary mode.
* `StringTextSerializer`: a serializer that serializes data into a string stream in text mode.
* `FileBinarySerializer`: a serializer that serializes data into a file stream in binary mode.
* `FileTextSerializer`: a serializer that serializes data into a file stream in text mode.

The implementation of these serialization classes are just template instantiations of the underlying template class `template < typename P > class Serializer`. The following table contains the equivalent declaration statements of these classes.

| Name (in `thunder` namespace)   | Declaration (in `thunder::serializer` nemaspace)   |
|---------------------------------|-----------------------------------------------------|
| StringBinarySerializer          | Serializer< BinaryProtocol< ::std::stringstream > > |
| StringTextSerializer            | Serializer< TextProtocol< ::std::stringstream > >   |
| FileBinarySerializer            | Serializer< BinaryProtocol< ::std::fstream > >      |
| FileTextSerializer              | Serializer< TextProtocol< ::std::fstream > >        |

The `thunder::serializer` namespace contains the actual implementation of these classes. The template class `Serializer` is used to handle serialization logic complexities such as avoiding duplicated pointer data saving, and finding the correct underlying dynamically binded types for pointers of classes in the inheritance hierarchy. After figuring out whether data should really be saved and the actual underlying type for dynamic binded pointers, it then delegates the actual serialization process to the associated protocol class.

The template classes `BinaryProtocol` and `TextProtocol` are two protocol classes that define how to serialize basic data types, and call user-defined serialization methods if the data is not a basic data type. As their names suggest, `BinaryProtocol` serializes basic data types into sequences of bytes using their representations in memory directly; whereas `TextProtocol` serializes them into text strings. Depending on the stream class used, the protocol classes can output to either string streams (via `std::stringstream`) or file streams (via `std::fstream`).

As a user, you do not need to bother with these details, no matter you are just using Thunder-Serializer for serializing Thunder data types, or making your own classes serializable through Thunder-Serializer. Using the four explicit serializer classes above is enough for normal uses.

### Symbols defined in Serializer

| Symbol               | Description                                                                                       |
|----------------------|---------------------------------------------------------------------------------------------------|
| protocol_type        | The protocol class type used to specialize the serializer class.                                  |
| stream_type          | The stream class type used to specialize the protocol class associated with the serializer class. |
| openmode             | The type for stream opening mode.                                                                 |

### Constants defined in Serializer

`Serializer` contains the following constants for specifying file open modes via the `|` operator. They are the same as those in [`::std::ios_base::openmode`](http://en.cppreference.com/w/cpp/io/ios_base/openmode).

| Constant  | Description                                      |
|-----------|--------------------------------------------------|
| app       | Seek to the end of stream before each write      |
| binary    | Open in binary mode                              |
| in        | Open for reading                                 |
| out       | Open for writing                                 |
| trunc     | Discard the contents of the stream when opening  |
| ate       | Seek to the end of stream immediately after open |

For example

```c++
FileBinarySerializer serializer("/tmp/file.tdb", FileBinarySerializer::in | FileBinarySerializer::out);
```

### template < typename... G > Serializer(G... g);

Construct a new serializer instance with parameter pack `g` delegated to the constructor of its protocol, which is in turn delegated to the constructor of the stream class used to specialize either `BinaryProtocol` or `TextProtocol`.

The following constructors are template-instantiated for string streams and can be used with `StringBinarySerializer` or `StringTextSerializer`.

| Template Specialization                                              | Description                                                                                                                |
|----------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| Serializer()                                                         | Default constructor. An empty string stream is created with stream mode `in | out`.                                         |
| Serializer(openmode mode)                                            | Construct the serializer using an empty string stream and set the stream mode.                                              |
| Serializer(const ::std::string &str)                                 | Construct the serializer by creating a string stream copying the contents of the given string with stream mode `in  | out`. |
| Serializer(const ::std::string &str, openmode mode)                  | Construct the serializer by creating a string stream copying the contents of the given string and setting the stream mode.  |

The following constructors are template-instantiated for file streams and can be used with `FileBinarySerializer` or `FileTextSerializer`.

| Template Specialization                                              | Description                                                                                                                                                 |
|----------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Serializer()                                                         | Default constructor. A stream that is not associated with a file is created. One must obtain the stream and open a file with it before using the serializer object. |
| Serializer(const char *filename)                                     | Construct the serializer by opening a file with stream mode `in | out`.                                                                                             |
| Serializer(const char *filename, openmode mode)                      | Construct the serializer by opening a file and setting the stream mode.                                                                                             |
| Serializer(const ::std::string &filename)                            | Construct the serializer by opening a file with strea mode `in | out`..                                                                                             |
| Serializer(const ::std::string &str, openmode mode)            | Construct the serializer by opening a file and setting the stream mode.                                                                                             |

### Serializer(const Serializer &s) = delete;

Disabled copy-constructor.

### Serializer& operator=(Serializer s) = delete;

Disabled assignment operator.

### const protocol_type& protocol() const;
### protocol_type& protocol();

Obtain the underlying protocol associated with the serializer.

### template < typename T > void save(const T &t);

Save object `t` of type `T` in the serializer.

### template < typename T > void load(T *t);

Load object of type `T` to the location pointed to by `t`.

### template < typename T > void save(T* const &t);

Save the object of type `T` at the location pointed to by `t`. If the object at location `t` is already saved, it will reuse previously saved space to avoid duplicated saving.

### template < typename T > void load(T* *t);

Load the object of type `T` and make `*t` point to it. If the object was already loaded, it will make `*t` point to the previous location to avoid duplicated loading.

### template < typename T > void save(const ::std::shared_ptr< T > &t);

Save the object of type `T` at the location pointed to by `t`. If the object at location `t` is already saved, it will reuse previously saved space to avoid duplicated saving.

### template < typename T > void load(::std::shared_ptr< T > *t);

Load the object of type `T` and make `*t` a shared pointer to it. If the object was already loaded as a `::std::shared_ptr< T >`, it will point to the same location as before and share the pointer.

## Make your own classes serializable

If you want your class to be serializable using Thunder-Serializer, you only need to overload the template functions `save` and `load` in the `thunder::serialize` namespace. For example

```c++
#include <string>
#include <thunder/serializer.hpp>

class Person {
 public:
  Person(int age = 0, const ::std::string &name = "") :
      age_(age), name_(name) {}
  int age() const { return age_; }
  ::std::string name() const {return name_;}
 private:
  int age_;
  ::std::string name_;
};

namespace thunder {
namespace serializer {

template < typename S >
void save(S *s, const Person &t) {
  s->save(t.age());
  ::std::string name = t.name();
  s->save(name.length());
  for (int i = 0; i < name.length(); ++i) {
    s->save(name[i]);
  }
}
template < typename S >
void load(S *s, Person *t) {
  int age;
  s->load(&age);
  typename ::std::string::size_type length;
  s->load(&length);
  ::std::string name(length, '\0');
  for (int i = 0; i < length; ++i) {
    s->load(&name[i]);
  }
  *t = Person(age, name);
}

}  // namespace serializer
}  // namespace thunder
```

Then, you can use serialization in the following way

```c++
#include <cstdio>
#include <thunder/serializer.hpp>
#include "person.hpp"

int main() {
  ::thunder::StringTextSerializer string_serializer;
  Person person(18, "Bob");
  ::std::printf("Saving for person age = %d, name = %s\n",
                person.age(), person.name().c_str());
  string_serializer.save(person);
  ::std::printf("Saved person data = %s\n",
                  string_serializer.protocol().stream().str().c_str());
  Person person_loaded;
  string_serializer.load(&person_loaded);
  ::std::printf("Loaded person age = %d, name = %s\n",
                person_loaded.age(), person.name().c_str());
}
```

The output looks like this

```
Saving for person age = 18, name = Bob
Saved person data = 18 3 66 111 98
Loaded person age = 18, name = Bob
```

Note that in the implementation above you have to implement string serialization one character at a time because currently Thunder-Serialization only supports limited C++ standard library data types. The work on implementing more of them is under-going.

## Internal details in thunder::serialize namespace

Internally, Thunder-Serializer is implemented via the following templated classes and functions.

### template < typename P > class Serializer;

The `Serializer` class handles the logic of serialization, such as dynamic binding, pointer semantics and shared pointer serialization. The template parameter `P` is an instantiation of either `BinaryProtocol` or `TextProtocol` class. For its symbols, constants and member functions, please refer to the previous sections.

Other than these serialization logic, the `Serializer` class does not actually save data. Rather, it delegate actual serialization of data to either `BinaryProtocol` or `TextProtocol`, depending on which is used for the template instantiation.

### template < typename M > class BinaryProtocol;

The protocol class that serializes data in binary format. It dumps basic data types as a sequence of bytes in memory, therefore the serialization result may not be compatible with a different machine architecture.

#### Symbols defined in BinaryProtocol

| Symbol               | Description                                                                                       |
|----------------------|---------------------------------------------------------------------------------------------------|
| stream_type          | The stream class type used to instantiate the protocol class.                                     |
| char_type            | The type of character that the stream accepts as basic units.                                     |
| openmode             | The type for stream opening mode.                                                                 |

#### Constants defined in BinaryProtocol

`BinaryProtocol` contains the following constants for specifying file open modes via the `|` operator. They are the same as those in [`::std::ios_base::openmode`](http://en.cppreference.com/w/cpp/io/ios_base/openmode).

| Constant  | Description                                      |
|-----------|--------------------------------------------------|
| app       | Seek to the end of stream before each write      |
| binary    | Open in binary mode                              |
| in        | Open for reading                                 |
| out       | Open for writing                                 |
| trunc     | Discard the contents of the stream when opening  |
| ate       | Seek to the end of stream immediately after open |

#### template < typename... G > explicit BinaryProtocol(G... g);

Construct a new `BinaryProtocol` instance with parameter pack `g` delegated to the constructor of its stream class.

#### BinaryProtocol(const BinaryProtocol &p) = delete;

Disabled copy-constructor.

#### BinaryProtocol& operator=(BinaryProtocol p) = delete;

Disabled assignment operator.

#### const stream_type &stream() const;
#### stream_type & stream();

Get the stream associated with the protocol.

#### template < typename S, typename T > void save(S *s, const T &t);

Save some data of type `T` to serializer `s`. If `T` is a basic data type, it puts the underlying data into the associated stream. Otherwise, it calls the static function `::thunder::serializer::save(s, t)`.

As a result, to implement saving for your own class, you only need to implement a template specialization of the function `::thunder::serializer::save(s, t)`.

#### template < typename S, typename T > void load(S *s, T *t);

Load some data of type `T` from serializer `s`. If `T` is a basic data type, it reads the underlying data from the associated stream. Otherwise, it calls the static function `::thunder::serializer::load(s, t)`.

As a result, to implement loading for your own class, you only need to implement a template specialization of the function `::thunder::serializer::load(s, t)`.

#### Basic data types

The following basic data types are saved and loaded directly in `BinaryProtocol`.

* `char`
* `signed char`
* `unsigned char`
* `wchar_t`
* `char16_t`
* `char32_t`
* `short`
* `unsigned short`
* `int`
* `unsigned int`
* `long`
* `unsigned long`
* `long long`
* `unsigned long long`
* `float`
* `double`
* `long double`

### template < typename M > class TextProtocol;

The protocol class that serializes data in text format. It dumps basic data types as a sequence of bytes in memory, therefore the serialization result may not be compatible with a different machine architecture.

#### Symbols defined in TextProtocol

| Symbol               | Description                                                                                       |
|----------------------|---------------------------------------------------------------------------------------------------|
| stream_type          | The stream class type used to instantiate the protocol class.                                     |
| char_type            | The type of character that the stream accepts as basic units.                                     |
| openmode             | The type for stream opening mode.                                                                 |

#### Constants defined in TextProtocol

`TextProtocol` contains the following constants for specifying file open modes via the `|` operator. They are the same as those in [`::std::ios_base::openmode`](http://en.cppreference.com/w/cpp/io/ios_base/openmode).

| Constant  | Description                                      |
|-----------|--------------------------------------------------|
| app       | Seek to the end of stream before each write      |
| binary    | Open in binary mode                              |
| in        | Open for reading                                 |
| out       | Open for writing                                 |
| trunc     | Discard the contents of the stream when opening  |
| ate       | Seek to the end of stream immediately after open |

#### template < typename... G > explicit TextProtocol(G... g);

Construct a new `TextProtocol` instance with parameter pack `g` delegated to the constructor of its stream class.

#### TextProtocol(const TextProtocol &p) = delete;

Disabled copy-constructor.

#### TextProtocol& operator=(TextProtocol p) = delete;

Disabled assignment operator.

#### const stream_type &stream() const;
#### stream_type & stream();

Get the stream associated with the protocol.

#### template < typename S, typename T > void save(S *s, const T &t);

Save some data of type `T` to serializer `s`. If `T` is a basic data type, it puts the underlying data into the associated stream. Otherwise, it calls the static function `::thunder::serializer::save(s, t)`.

As a result, to implement saving for your own class, you only need to implement a template specialization of the function `::thunder::serializer::save(s, t)`.

#### template < typename S, typename T > void load(S \*s, T *t);

Load some data of type `T` from serializer `s`. If `T` is a basic data type, it reads the underlying data from the associated stream. Otherwise, it calls the static function `::thunder::serializer::load(s, t)`.

As a result, to implement loading for your own class, you only need to implement a template specialization of the function `::thunder::serializer::load(s, t)`.

#### Basic data types

The following basic data types are saved and loaded directly in `TextProtocol`.

* `char`
* `signed char`
* `unsigned char`
* `wchar_t`
* `char16_t`
* `char32_t`
* `short`
* `unsigned short`
* `int`
* `unsigned int`
* `long`
* `unsigned long`
* `long long`
* `unsigned long long`
* `float`
* `double`
* `long double`

### template < typename S, typename T > void save(S *s, const T &t);
### template < typename S, typename T > void load(S \*s, T *t);

Static functions used by `BinaryProtocol` or `TextProtocol` to serialize non-basic classes. One can specialize the class type `T` of these functions to implement serialization for their own classes.

#### Specialized standard template classes

The following template classes have their `save` and `load` functions implemented in Thunder-Serializer. More will be implemented in the future.

* `template < typename T > ::std::complex< T >`
* `template < typename T1, typename T2 > ::std::pair< T1, T2>`
