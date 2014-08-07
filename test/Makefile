# List all the test targets here
targets = storage

# Define useful flags
CXXFLAGS += -std=c++11 -Wall -I..
LDFLAGS += -lgtest -lgtest_main

# Create an implicit rule on test
.PHONY : %.test
%.test : %
	@echo Testing: $^
	./$^

# This is the default target
.PHONY : all
all: $(targets)
	@echo Run \"make test\" to perform test

# This is the target rule
$(targets) : % : %.cpp

# This is the test rule
test : $(patsubst %,%.test,$(targets))
	@echo Testing completed.

# Clean simply cleans targets
.PHONY : clean
clean :
	-rm $(targets)
