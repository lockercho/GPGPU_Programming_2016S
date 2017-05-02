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

__global__ void initData(const char * text, int *gStart, int * gEnd, int text_size, int * nWords) {
    int begin =0, end = 0;
    int index = 0;
    for(int i=0 ; i< text_size ; i++) {
        if(text[i] != '\n') {
            end++;
        } else {
            if(begin != end) {
                gStart[index] = begin;
                gEnd[index] = begin;
                index++;
            } 
            begin = end = end+1;
        }
    }
    if(begin != end) {
        gStart[index] = begin;
        gEnd[index] = begin;     
        index++;
    }
    *nWords = index;
}

__global__ void fill(int * pos, int size, int val) {
    for(int i=0 ; i < size ; i++) {
        pos[i] = val;
    }
}

__global__ void sequence(int * start, int * end) {
    int count = end - start;
    for(int i=0 ; i<count ; i++) {
        *(start + i) = i + 1;
    }
}

__global__ void iterateIt(const char * text, int * pos, int * start, int * end, int nWords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < nWords) {
        int s = start[idx];
        int e = end[idx];   

        for(int i=s ; i<e ; i++) {
            *(pos + i) = i - s + 1;
        }
    }
}

void CountPosition2(const char *text, int *pos, int text_size)
{
    int * start;
    int * end;
    int * nWords;

  	cudaMalloc(&nWords, sizeof(int));
    // cudaMalloc a device array
    cudaMalloc((void**)&start, text_size); 
    cudaMalloc((void**)&end, text_size); 

	initData<<<1,1>>>(text, start, end, text_size, nWords);
    
    int blockSize = 8;
    int nBlock = nWords / blockSize + (nWords % blockSize == 0 ? 0: 1);

    fill<<<1,1>>>(pos, text_size, 0);
    iterateIt<<<nBlock, blockSize>>>(text, pos, start, end, nWords);
}


