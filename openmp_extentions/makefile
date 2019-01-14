all:
	swig -c++ -python roi_avg.i
	python setup.py build_ext --inplace
roi:
	g++ -c -fpic roi_avg_wrap.cxx roi_avg.cpp -I/usr/include/python2.7  -std=c++11 -fopenmp

shared:
	g++ -shared roi_avg.o roi_avg_wrap.o -o _roi_avrg.so

clean:
	rm -f *.o
	rm -f roi_avg.py
	rm -f *.cxx
	rm -f *.so
	rm -r build
