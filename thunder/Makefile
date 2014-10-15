# List all the libraries here
libs = libthunder_exception.so libthunder_storage.so libthunder_tensor.so

# List all headers that will be installed here
hdrs = exception.hpp exception/exception.hpp exception/contiguity_error.hpp \
	storage.hpp storage/storage.hpp storage/storage-inl.hpp

# Set the location of installation here
PREFIX ?= /usr/local

# Complilation flags
CXXFLAGS += -std=c++11 -O3 -Wall -I.. -fPIC

# Loader flags
LDFLAGS += -shared -L.

# The all PHONY target is used for compilation commands
.PHONY : all
all: $(libs)
	@echo Run \"make install\" to install the library

# The install PHONY target is used for executing installation commands
.PHONY : install
install : $(libs)
	@echo Install command is not yet available

# The clean PHONY target is used for cleanup the compilation
.PHONY : clean
clean :
	-rm $(libs)
	-rm exception/*.o
	-rm storage/*.o
	-rm tensor/*.o

# The exception library
libthunder_exception.so : exception/contiguity_error.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

# The storage library
libthunder_storage.so : storage/storage.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

# The tensor library
tensor_objs = tensor/tensor.o
tensor_ldflags = -lthunder_exception -lthunder_storage
libthunder_tensor.so : $(tensor_objs) libthunder_exception.so libthunder_storage.so
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $(tensor_ldflags) $(tensor_objs) -o $@
