# Thunder-Exception

The Thunder-Exception package is a package used by all other Thunder packages for error handling. We require that all thunder packages only use these exception classes. These classes will be available if you include the thunder/exception.hpp header.

# class thunder::exception::exception;

A typedef of [std::exception](http://en.cppreference.com/w/cpp/error/exception).

## exception();

Default construct in which what() returns empty string.

## exception(const exception &other);

Copy construct. Initializes the contents with those of other.

### Parameters

* other: another exception to assign the contents of.

## virtual ~exception();

Destroys the exception object.

## exception& operator=(const exception& other);

Copy assigment operator. Assigns the contents of other.

### Parameters

* other: another exception to assign the contents of.

## virtual const char* what() const;

Returns the explanatory string.

### Return value

Pointer to a null-terminated string with explanatory information. The pointer is guaranteed to be valid at least until the exception object from which it is obtained is destroyed, or until a non-const member function on the exception object is called.

# class thunder::logic_error;

A typedef of [std::logic_error](http://en.cppreference.com/w/cpp/error/logic_error). It reports errors that are a consequence of faulty logic within the program such as violating logical preconditions or class invariants and may be preventable.

No Thunder components should throw this exception directly, but the exception types thunder::invalid_argument, thunder::domain_error, thunder::length_error, thunder::out_of_range andthunder::contiguity_error are derived from thunder::logic_error.

## explicit logic_error(const std::string& what_arg );
## explicit logic_error(const char* what_arg );

Constructs the exception object with what_arg as explanatory string that can be accessed through what().

### Parameters

* what_arg: explanatory string

## Inherited from ::thunder::exception::exception

| Member functions                  | Description                     |
|-----------------------------------|---------------------------------|
| virtual ~exception();             | Destructs the exception object. |
| virtual const char* what() const; | Returns the explanatory string. |

Returns the explanatory string.

# class thunder::runtime_error;

A typedef of [std::runtime_error](http://en.cppreference.com/w/cpp/error/runtime_error). It reports errors that are due to events beyond the scope of the program and can not be easily predicted.

No Thunder components should throw this exception directly, but the exception types thunder::range_error, thunder::overflow_error, thunder::underflow_error, andthunder::system_error are derived from thunder::runtime_error.

## explicit runtime_error(const std::string& what_arg );
## explicit runtime_error(const char* what_arg );

Constructs the exception object with what_arg as explanatory string that can be accessed through what().

### Parameters

* what_arg: explanatory string

## Inherited from ::thunder::exception::exception

| Member functions                  | Description                     |
|-----------------------------------|---------------------------------|
| virtual ~exception();             | Destructs the exception object. |
| virtual const char* what() const; | Returns the explanatory string. |


# class thunder::invalid_argument;

A typedef of [std::invalid_argument](http://en.cppreference.com/w/cpp/error/invalid_argument). It reports errors that arise because an argument value has not been accepted.

## explicit invalid_argument(const std::string& what_arg );
## explicit invalid_argument(const char* what_arg );

Constructs the exception object with what_arg as explanatory string that can be accessed through what().

### Parameters

* what_arg: explanatory string

## Inherited from ::thunder::exception::exception

| Member functions                  | Description                     |
|-----------------------------------|---------------------------------|
| virtual ~exception();             | Destructs the exception object. |
| virtual const char* what() const; | Returns the explanatory string. |


# class thunder::domain_error;

A typedef of [std::domain_error](http://en.cppreference.com/w/cpp/error/domain_error). It may be used by the implementation to report domain errors, that is, situations where the inputs are outside of the domain on which an operation is defined.

## explicit domain_error(const std::string& what_arg );
## explicit domain_error(const char* what_arg );

Constructs the exception object with what_arg as explanatory string that can be accessed through what().

### Parameters

* what_arg: explanatory string

## Inherited from ::thunder::exception::exception

| Member functions                  | Description                     |
|-----------------------------------|---------------------------------|
| virtual ~exception();             | Destructs the exception object. |
| virtual const char* what() const; | Returns the explanatory string. |

# class thunder::length_error;

A typedef of [std::length_error](http://en.cppreference.com/w/cpp/error/length_error). It reports errors that are consequence of attempt to exceed implementation defined length limits for some object.

## explicit length_error(const std::string& what_arg );
## explicit length_error(const char* what_arg );

Constructs the exception object with what_arg as explanatory string that can be accessed through what().

### Parameters

* what_arg: explanatory string

## Inherited from ::thunder::exception::exception

| Member functions                  | Description                     |
|-----------------------------------|---------------------------------|
| virtual ~exception();             | Destructs the exception object. |
| virtual const char* what() const; | Returns the explanatory string. |

# class thunder::out_of_range;

A typedef of [std::out_of_range](http://en.cppreference.com/w/cpp/error/out_of_range). It reports errors that are consequence of attempt to access elements out of defined range.

## explicit out_of_range(const std::string& what_arg );
## explicit out_of_range(const char* what_arg );

Constructs the exception object with what_arg as explanatory string that can be accessed through what().

### Parameters

* what_arg: explanatory string

## Inherited from ::thunder::exception::exception

| Member functions                  | Description                     |
|-----------------------------------|---------------------------------|
| virtual ~exception();             | Destructs the exception object. |
| virtual const char* what() const; | Returns the explanatory string. |

# class thunder::range_error;

A typedef of [std::range_error](http://en.cppreference.com/w/cpp/error/range_error). It can be used to report range errors (that is, situations where a result of a computation cannot be represented by the destination type).

## explicit range_error(const std::string& what_arg );
## explicit range_error(const char* what_arg );

Constructs the exception object with what_arg as explanatory string that can be accessed through what().

### Parameters

* what_arg: explanatory string

## Inherited from ::thunder::exception::exception

| Member functions                  | Description                     |
|-----------------------------------|---------------------------------|
| virtual ~exception();             | Destructs the exception object. |
| virtual const char* what() const; | Returns the explanatory string. |

# class thunder::overflow_error;

A typedef of [std::overflow_error](http://en.cppreference.com/w/cpp/error/overflow_error). It can be used to report arithmetic overflow errors (that is, situations where a result of a computation is too large for the destination type).

## explicit overflow_error(const std::string& what_arg );
## explicit overflow_error(const char* what_arg );

Constructs the exception object with what_arg as explanatory string that can be accessed through what().

### Parameters

* what_arg: explanatory string

## Inherited from ::thunder::exception::exception

| Member functions                  | Description                     |
|-----------------------------------|---------------------------------|
| virtual ~exception();             | Destructs the exception object. |
| virtual const char* what() const; | Returns the explanatory string. |

# class thunder::underflow_error;

A typedef of [std::underflow_error](http://en.cppreference.com/w/cpp/error/underflow_error). It may be used to report arithmetic underflow errors (that is, situations where the result of a computation is a subnormal floating-point value).

## explicit underflow_error(const std::string& what_arg );
## explicit underflow_error(const char* what_arg );

Constructs the exception object with what_arg as explanatory string that can be accessed through what().

### Parameters

* what_arg: explanatory string

## Inherited from ::thunder::exception::exception

| Member functions                  | Description                     |
|-----------------------------------|---------------------------------|
| virtual ~exception();             | Destructs the exception object. |
| virtual const char* what() const; | Returns the explanatory string. |

# class thunder::contiguity_error;

A type of exception thrown when the data (such as a multi-dimensional array) is does not meet the contiguity requirement.

## explicit contiguity_error(const std::string& what_arg );
## explicit contiguity_error(const char* what_arg );

Constructs the exception object with what_arg as explanatory string that can be accessed through what().

### Parameters

* what_arg: explanatory string

## Inherited from ::thunder::exception::exception

| Member functions                  | Description                     |
|-----------------------------------|---------------------------------|
| virtual ~exception();             | Destructs the exception object. |
| virtual const char* what() const; | Returns the explanatory string. |
