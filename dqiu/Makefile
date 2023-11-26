.PHONY: all libeasynn easynn_test ## playground

all: libeasynn easynn_test ## playground

libeasynn:
	#g++ -Wall src/*.cpp -fPIC -O -g -shared -o libeasynn.so
	g++ -Wall -std=gnu++11 src/*.cpp -fPIC -O -g -shared -o libeasynn.so

easynn_test: libeasynn
	#g++ -Wall easynn_test.cpp -g -lm -L. -Wl -rpath=. -leasynn -o easynn_test
	g++ -std=gnu++11 -Wall easynn_test.cpp -g -lm -L. -leasynn -o easynn_test

# playground:
# 	g++ -Wall -Wc++11-extensions src/playground.cpp -g -lm -L. -Wl -o playground

