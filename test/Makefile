# List all the test targets here
targets = storage index_iterator tensor_access tensor_modify tensor_type \
	tensor_unary tensor_binary tensor_ternary tensor_reduction \
	tensor_static tensor_operator tensor_serialize complex_unary \
	complex_binary complex_ternary complex_reduction complex_operator

CXXFLAGS += -std=c++11 -Wall -I..
LDFLAGS += -L../thunder -lgtest -lgtest_main -lboost_serialization \
	-lthunder_exception -lthunder_storage -lthunder_tensor

.PHONY : all
all: $(targets)
	@echo Run \"make test\" to perform test

.PHONY : test
test : $(targets)
	-count=0; \
	success=0; \
	failure=0; \
	for target in $(targets); do \
		let count+=1; \
		echo "Testing $$count/$(words $(targets)): $$target"; \
		./$$target; \
		if [ $$? == 0 ]; then \
			let success+=1; \
		else \
			let failure+=1; \
		fi; \
		echo; \
	done; \
	echo \
	"TOTAL: $(words $(targets)) | SUCCESS: $$success | FAILURE: $$failure";

.PHONY : clean
clean :
	-rm $(targets)

$(targets) : % : %.cpp
