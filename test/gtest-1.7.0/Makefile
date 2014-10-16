# List all the test targets here
targets = libgtest.a libgtest_main.a
objects = gtest/gtest-all.o gtest/gtest_main.o
CXXFLAGS += -I. -g -Wall -Wextra
LDFLAGS +=

.PHONY : all
all: $(targets)

.PHONY : clean
clean :
	-$(RM) $(targets) $(objects)

$(objects) : %.o : %.cc

libgtest.a : gtest/gtest-all.o
	$(AR) rs $@ $^

libgtest_main.a : gtest/gtest-all.o gtest/gtest_main.o
	$(AR) rs $@ $^
