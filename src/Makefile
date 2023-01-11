CC = gcc
NVCC = nvcc


all: cpu_sample.out cuda_sample.out

cpu_sample.out: cpu_sample.c
	$(CC) -o $@ $<

cuda_sample.out: cuda_sample.c
	$(NVCC) -o $@ $<

.PHONY: clean
clean:
	rm -f cpu_sample.out cuda_sample.out
