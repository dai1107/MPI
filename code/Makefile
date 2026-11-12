CXX = mpic++
CXXFLAGS = -O3 -std=c++17 -Wall
LDFLAGS =

all: main

main: main.cpp main.h
	$(CXX) $(CXXFLAGS) main.cpp -o main $(LDFLAGS)

clean:
	rm -f main

.PHONY: all clean
