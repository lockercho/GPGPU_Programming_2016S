#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
        const float *background,
        const float *target,
        const float *mask,
        float *output,
        const int wb, const int hb, const int wt, const int ht,
        const int oy, const int ox
        )
{
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
    const int xt = blockIdx.x * blockDim.x + threadIdx.x;
    const int curt = wt*yt+xt;
    if (yt < ht and xt < wt and mask[curt] > 127.0f) {
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
		float * fixed, float * target ,
        const float* mask, const float * border, 
        float * guess_prev, float * guess_next, 
        const int wt, const int ht) {

    int i_diff[] = {-1, 0, 0, 1};
    int j_diff[] = { 0,-1, 1, 0};
    for(int i=0; i<ht ; i++) {
        for(int j=0; j<wt ; j++) {
            if(mask[i*wt + j] != 255) 
                continue;
            for(int c=0 ; c<3 ; c++) {
                float sum1 = 0, sum2 = 0;
                int p = i * 3 * wt + j + c;
                // check is border
                for(int n=0; n<4 ; n++) {
                    int neighbor = (i+i_diff[n]) * 3 * wt + j + j_diff[n] + c;
                    int b_neighbor = i*wt + j;
                    if(neighbor < 0 || neighbor >= 3 * ht * wt) {
                        continue;
                    }
                    if(border[b_neighbor] == 1) {
                        sum1 += dest[neighbor];
                    } else {
                        sum1 += guess_prev[neighbor];
                    }
                    sum2 += target[p] - target[neighbor];
                }
                float newVal = (sum1 + sum2) / 4.f;

                // clamp
                guess_next[p] = min(255, max(0, newVal));
            }
        }
    }
}

__global__ void CalculateFixed(
		float * fixed, const float * background,
        const float * mask, float * border,
        const int wb, const int hb,
        const int wt, const int ht, 
        const int oy, const int ox) {

	// set border & initial guess
    for(int i=0 ; i<ht ; i++) {
        for(int j=0; j<wt ; j++) {
        	for(int c=0 ;c<3 ;c++) {
        		fixed[i*3*wt+j+c] = background[(i+oy)*3*wt+(j+ox)+c]; 
        	}
            // check is interior
            bool is_interior = true;
            // check top
            if(i-1 < 0 || mask[(i-1)*wt+j] != 255) {
                border[i*wt+j] = 1;
                continue;
            }
            // check bottom
            if(is_interior && (i+1 >= ht || mask[(i+1)*wt+j] != 255)) {
                border[i*wt+j] = 1;
                continue;
            }
            // check left
            if(is_interior && (j-1 < 0 || mask[(i)*wt + j-1] != 255)) {
                border[i*wt+j] = 1;
                continue;
            }
            // check right
            if(is_interior && (j+1 >= wt || mask[i*wt+j+1] != 255)) {
                border[i*wt+j] = 1;
                continue;
            }
        }
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
    fprintf(stderr, "wb: %d hb: %d, wt: %d, ht: %d, oy: %d, ox: %d\n", wb,hb,wt,ht,oy,ox);

    // set up
    float *fixed, *guess_prev, *guess_next, *border;
    cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
    cudaMalloc(&guess_prev, 3*wt*ht*sizeof(float));
    cudaMalloc(&guess_next, 3*wt*ht*sizeof(float));
    cudaMalloc(&border, wt*ht*sizeof(int));
    // initialize the iteration
    dim3 gdim(CeilDiv(wt,32), CeilDiv(ht,16)), bdim(32,16);
    CalculateFixed<<<gdim, bdim>>>(fixed, background, mask, border, wb, hb, wt, ht, oy, ox);
    cudaMemcpy(guess_prev, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);
    cudaMemcpy(guess_next, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);
    // iterate
    for(int i=0;i<10000;++i) { 
        PoissonImageCloningIteration<<<gdim, bdim>>>(
        	fixed, target, mask, border, 
        	guess_prev, guess_next, wt, ht);
        // use last next as new prev
        PoissonImageCloningIteration<<<gdim, bdim>>>(
        	fixed, target, mask, border, 
        	guess_next, guess_prev, wt, ht);
    }
    // copy the image back
    cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
    SimpleClone<<<gdim, bdim>>>(
        background, guess_next, mask, output,
        wb, hb, wt, ht, oy, ox);
    // clean up
    cudaFree(fixed);
    cudaFree(buf1);
    cudaFree(buf2);
}

