# List all the packages here
pkgs = exception storage tensor
libs = $(addsuffix .so, $(addprefix libthunder_, $(pkgs)))
# Set the location of installation here
prefix ?= /usr/local
exec_prefix ?= $(prefix)
includedir ?= $(prefix)/include
libdir ?= $(exec_prefix)/lib

# Compliler flags
CXXFLAGS += -std=c++11 -O3 -Wall -I.. -fPIC
# Loader flags
LDFLAGS += -shared -L.

# The 'all' PHONY target is used for default compilation
.PHONY : all
all: $(libs)
	@echo Run \"make install\" to install the library

# The 'install' PHONY target is used for executing installation commands
.PHONY : install
install : $(libs)
	test -d $(prefix) || mkdir $(prefix)
	test -d $(includedir) || mkdir $(includedir)
	test -d $(includedir)/thunder || mkdir $(includedir)/thunder
	for hdr in *.hpp; do \
		install -m 0644 $$hdr $(includedir)/thunder; \
	done
	for pkg in $(pkgs); do \
		test -d $(includedir)/thunder/$$pkg || mkdir $(includedir)/thunder/$$pkg; \
		for hdr in $$pkg/*.hpp; do \
			install -m 0644 $$hdr $(includedir)/thunder/$$pkg; \
		done; \
	done
	test -d $(libdir) || mkdir $(libdir)
	for lib in $(libs); do \
		install -m 0755 $$lib $(libdir); \
	done

# The 'clean' PHONY target is used for cleanup the compiled object code
.PHONY : clean
clean :
	-$(RM) $(libs)
	-for pkg in $(pkgs); do $(RM) $$pkg/*.o; done

# The exception library
exception_objs = exception/contiguity_error.o
exception_deps =
exception_ldflags =
libthunder_exception.so : $(exception_objs) $(addsuffix .so, $(addprefix libthunder_, $(exception_deps)))
	$(CXX) $(LDFLAGS) $(addprefix -lthunder_, $(exception_deps)) $(exception_objs) $(exception_ldflags) -o $@

# The storage library
storage_objs = storage/storage.o
storage_deps =
storage_ldflags = -lboost_serialization
libthunder_storage.so : $(storage_objs) $(addsuffix .so, $(addprefix libthunder_, $(storage_deps)))
	$(CXX) $(LDFLAGS) $(addprefix -lthunder_, $(storage_deps)) $(storage_objs) $(storage_ldflags) -o $@

# The tensor library
tensor_objs = tensor/tensor.o
tensor_deps = exception storage
tensor_ldflags = -lboost_serialization
libthunder_tensor.so : $(tensor_objs) $(addsuffix .so, $(addprefix libthunder_, $(tensor_deps)))
	$(CXX) $(LDFLAGS) $(addprefix -lthunder_, $(tensor_deps)) $(tensor_objs) $(tensor_ldflags) -o $@
