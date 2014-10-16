.PHONY : all
all:
	cd thunder && $(MAKE) all
	cd test && $(MAKE) all

.PHONY : install
install :
	cd thunder && $(MAKE) install

.PHONY : clean
clean :
	cd thunder && $(MAKE) clean
	cd test && $(MAKE) clean

.PHONY : test
test :
	cd thunder && $(MAKE)
	cd test && $(MAKE) test
