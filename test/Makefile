# List all the test targets here
targets = storage

CXXFLAGS += -std=c++11 -Wall -I..
LDFLAGS += -lgtest -lgtest_main

.PHONY : all
all: $(targets)
	@echo Run \"make test\" to perform test

.PHONY : test
test : $(targets)
	-for target in $(targets); do ./$$target; done

.PHONY : clean
clean :
	-rm $(targets)

$(targets) : % : %.cpp

