#include "lab3.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include "SyncedMemory.h"
#include "pgm.h"
using namespace std;

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
        const float *background,
        const float *target,
        const float *mask,
        float * border,
        float *output,
        const int wb, const int hb, const int wt, const int ht,
        const int oy, const int ox
        )
{
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
    const int xt = blockIdx.x * blockDim.x + threadIdx.x;
    const int curt = wt*yt+xt;
    if (yt < ht and xt < wt and mask[curt] > 127.0f and border[curt] < 127.0f) {
        const int yb = oy+yt, xb = ox+xt;
        const int curb = wb*yb+xb;
        if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
            output[curb*3+0] = target[curt*3+0];
            output[curb*3+1] = target[curt*3+1];
            output[curb*3+2] = target[curt*3+2];
        }
    }
}

__global__ void PoissonImageCloningIteration(
		float *fixed, const float *target ,
        const float* mask, float * border, 
        float * guess_prev, float * guess_next, 
        const int wt, const int ht) {

	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
    const int xt = blockIdx.x * blockDim.x + threadIdx.x;
    const int curt = wt*yt+xt;
	
	for(int c=0; c<3 ; c++) {
	    if(mask[curt] > 127 && border[curt] < 127) {

		    int i_diff[] = {-1, 0, 0, 1};
		    int j_diff[] = { 0,-1, 1, 0};
			
			float sum1 = 0, sum2 = 0;
			float num = 0;
		    // check is border
		    for(int n=0; n<4 ; n++) {
		    	int y = yt + i_diff[n];
		        int x = xt + j_diff[n];
		    	int neighbor = y * wt + x;
				
		        num++;
		        if(border[neighbor] > 127) {
		        	// neighbor is border
		            sum1 += fixed[neighbor * 3 + c];
		        } else {
		            sum1 += guess_prev[neighbor * 3 + c];
		        }
		        sum2 += target[curt * 3 + c] - target[neighbor * 3 + c];
		    }
		    float newVal = (sum1 + sum2) / 4.0;

		    // clamp
		    guess_next[curt * 3 + c] = newVal;
		}
	}
}

__global__ void CalculateFixed(
		float * fixed, const float * background,
        const float * mask, float * mMask, float * border,
        const int wb, const int hb,
        const int wt, const int ht, 
        const int oy, const int ox) {

    	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
        const int xt = blockIdx.x * blockDim.x + threadIdx.x;
        const int curt = wt*yt+xt;

    	for(int c=0 ;c<3 ;c++) {
    		fixed[curt*3+c] = background[((yt+oy)*wb+(xt+ox))*3+c]; 
    	}

    	border[curt] = 0;
    	if(mask[curt] < 127.0) {
    		return;
    	}
        // check is interior
        // check top
        if(yt-1 <= 0 || mask[(yt-1)*wt+xt] < 127.0) {
            border[curt] = 255;
            // mMask[curt] = 0;
            return;
        }
        // check bottom
        if(yt+1 >= ht || mask[(yt+1)*wt+xt] < 127.0) {
            border[curt] = 255;
            // mMask[curt] = 0;
            return;
        }
        // check left
        if(xt-1 <= 0 || mask[yt*wt + xt-1] < 127.0) {
            border[curt] = 255;
            // mMask[curt] = 0;
            return;
        }
        // check right
        if(xt+1 >= wt || mask[yt*wt+xt+1] < 127.0) {
            border[curt] = 255;
            // mMask[curt] = 0;
            return;
        }

}

void PoissonImageCloning(
        const float *background,
        const float *target,
        const float *mask,
        float *output,
        const int wb, const int hb, const int wt, const int ht,
        const int oy, const int ox
        ){ 

    // set up
    float *fixed, *guess_prev, *guess_next, *border;
    cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
    cudaMalloc(&guess_prev, 3*wt*ht*sizeof(float));
    cudaMalloc(&guess_next, 3*wt*ht*sizeof(float));
    cudaMalloc(&border, wt*ht*sizeof(float));
    // initialize the iteration
    dim3 gdim(CeilDiv(wt,32), CeilDiv(ht,16)), bdim(32,16);

    float * mMask;
    cudaMalloc(&mMask, wt*ht*sizeof(float));
    cudaMemcpy(mMask, mask, sizeof(float)*wt*ht, cudaMemcpyDeviceToDevice);

    CalculateFixed<<<gdim, bdim>>>(fixed, background, mask, mMask, border, wb, hb, wt, ht, oy, ox);
    
    cudaMemcpy(guess_prev, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);

    // iterate
    for(int i=0;i<20000;++i) { 
        PoissonImageCloningIteration<<<gdim, bdim>>>(
        	fixed, target, mask, border, 
        	guess_prev, guess_next, wt, ht);
        // flip next & prev
        PoissonImageCloningIteration<<<gdim, bdim>>>(
        	fixed, target, mask, border, 
        	guess_next, guess_prev, wt, ht);
    }
    
    // copy the image back
    cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
    // fprintf(stderr, "SimpleClone\n");
    SimpleClone<<<gdim, bdim>>>(
        background, guess_next, mask, border, output,
        wb, hb, wt, ht, oy, ox);     

    // clean up
    cudaFree(fixed);
    cudaFree(guess_prev);
    cudaFree(guess_next);
    cudaFree(border);
}

