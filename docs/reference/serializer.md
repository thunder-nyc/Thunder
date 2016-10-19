# Thunder-Serializer

The Thunder-Serializer package is a package used by many other Thunder packages for serialization data to strings or files. The goal of Thunder-Serializer includes

* Serialization of all fundamental types in C++11.
* Preserving data sharing relationships by only saving pointer memory once.
* Dynamic serialization of pointers that bind polymorphic types.
* Providing a non-intrusive interface for user-defined classes.

The current Thunder-Serializer implementation is only experimental, in that polymorphic serialization is not yet implemented. However, further development of Thunder-Serializer will ensure backward compatibility of the interfaces.

## Using Thunder-Serializer

Thunder-Serializer package contains the the following four classes for serialization. They are available in the `::thunder` namespace if you include `thunder/serializer.hpp`.

* `StringBinarySerializer`: a serializer that serializes data into a string stream in binary mode.
* `StringTextSerializer`: a serializer that serializes data into a string stream in text mode.
* `FileBinarySerializer`: a serializer that serializes data into a file stream in binary mode.
* `FileTextSerializer`: a serializer that serializes data into a file stream in text mode.

The implementation of these serialization classes are just template instantiations of the underlying template class `Serializer`. The following table contains the equivalent declaration statements of these classes.

| Name (in `::thunder` namespace) | Declaration (in `::thunder::serializer` nemaspace)  |
|---------------------------------|-----------------------------------------------------|
| StringBinarySerializer          | Serializer< BinaryProtocol< ::std::stringstream > > |
| StringTextSerializer            | Serializer< TextProtocol< ::std::stringstream > >   |
| FileBinarySerializer            | Serializer< BinaryProtocol< ::std::fstream > >      |
| FileTextSerializer              | Serializer< TextProtocol< ::std::fstream > >        |

The `::thunder::serializer` namespace contains the actual implementation of these classes. The template class `Serializer` is used to handle serialization logic complexities such as avoiding duplicated pointer data saving, and finding the correct underlying dynamically binded types for pointers of classes in the inheritance hierarchy. After figuring out whether data should really be saved and the actual underlying type for dynamic binded pointers, it then delegates the actual serialization process to the associated protocol class.

The template classes `BinaryProtocol` and `TextProtocol` are two protocol classes that define how to serialize basic data types, and call user-defined serialization methods if the data is not a basic data type. As their names suggest, `BinaryProtocol` serializes basic data types into sequences of bytes using their representations in memory directly; whereas `TextProtocol` serializes them into text strings. Depending on the stream class used, the protocol classes can output to either string streams (via `::std::stringstream`) or file streams (via `::std::fstream`).

As a user, you do not need to bother with these details, no matter you are just using Thunder-Serializer for serializing Thunder data types, or making your own classes serializable through Thunder-Serializer. Using the four explicit serializer classes above is enough for normal uses.

## Make your own classes serializable

If you want your class to be serializable using Thunder-Serializer, you only need to overload the template functions `save` and `load` in the `::thunder::serialize` namespace. For example

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

### template < typename P > class Serializer

### template < typename M > class BinaryProtocol

### template < typename M > class TextProtocol

