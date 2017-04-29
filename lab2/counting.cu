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
    thrust::device_ptr<const char> txt_ptr = thrust::device_pointer_cast(text);
//    thrust::fill(dev_ptr, dev_ptr + text_size, (int) 0);
    
    int begin =0, end = 0;
    for(int i=0 ; i < text_size ; i++) {
        if(txt_ptr[i] != '\n') {
            end++;
        } else {
            if(begin != end) 
                thrust::sequence(dev_ptr + begin, dev_ptr + end, 1);
            begin = end = end+1;
        }
    }
    if(begin != end) thrust::sequence(dev_ptr + begin, dev_ptr + end, 1);
    pos = thrust::raw_pointer_cast(dev_ptr);
}

void CountPosition2(const char *text, int *pos, int text_size)
{
}

