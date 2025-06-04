COMMON_OPTIONS = -Wall -Wno-unused-value -O3 -march=skylake

all:
	gcc $(COMMON_OPTIONS) *.c -o hashtable.out