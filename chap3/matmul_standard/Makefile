default: build ;

build:
	nvcc -lcurand -o matrixMul matrixMul.cu

run : build
	./matrixMul

clean:
	rm -f matrixMul
