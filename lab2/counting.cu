#include "counting.h"
#include <cstdio>
#include <cassert>
#include <string>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

void CountPosition1(const char *text, int *pos, int text_size)
{
    std::cerr << "textsize: " << text_size << std::endl;
    // wrap with a device_ptr 
    thrust::device_ptr<int> dev_ptr(pos);
    thrust::device_ptr<const char> txt_ptr(text);
    
    int begin =0, end = 0;
    for(int i=0 ; i < text_size ; i++) {
        if(txt_ptr[i] != '\n') {
            end++;
        } else {
            if(begin != end) thrust::sequence(dev_ptr + begin, dev_ptr + end, 1);
            begin = end = end+1;
        }
    }
    if(begin != end) thrust::sequence(dev_ptr + begin, dev_ptr + end, 1);
}

__global__ void fill(int * pos, int size, int val) {
	for(int i=0 ; i < size ; i++) {
		pos[i] = val;
	}
}

__global__ void sequence(int * start, int * end) {
	int i = 0;
	int count = end - start;
	for(int i=0 ; i<count ; i++) {
		*(start + i) = i + 1;
	}
}

__global void iterateIt(const char * text, int pos, int text_size) {
	for(int i=0 ; i< text_size ; i++) {
		if(text[i] != '\n') {
            end++;
        } else {
            if(begin != end) {
            	sequence<<<1,1>>>(pos+begin, pos+end);
            } 
            begin = end = end+1;
        }
	}
	if(begin != end) {
		sequence<<<1,1>>>(pos+begin, pos+end);
	}
}

void CountPosition2(const char *text, int *pos, int text_size)
{
	fill<<<1,1>>>(pos, text_size, 0);
	iterateIt<<<1,>>>(text, pos, text_size);
}

