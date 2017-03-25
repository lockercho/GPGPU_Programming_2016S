#include "lab1.h"
static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 24;

struct Lab1VideoGenerator::Impl {
    int t = 0;
};

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
    perlin = new Perlin();
    noiseArr = new float[W*H];
    // pre-generate all value
    float max = -1.0, min = 1.0;
    for(int i=0 ; i<W ; i++) {
        for(int j=0 ; j<H ; j++) {
            float tmp = perlin->noise(float(i) / W, float(j) / H, 0.24);
            noiseArr[j*W+i] = tmp;
            if (tmp < min) min = tmp;
            if (tmp > max) max = tmp;
        }        
    }
    float tmp = 1.0f / (max - min);
    // stretch noise to 0~1
    for(int i=0 ; i<W ; i++) {
        for(int j=0 ;j<H ; j++) {
            float n = noiseArr[j*W + i];
            n = -1.0f + 2.0f * (n - min)*tmp;
            // Remap to RGB friendly colour values in range between 0 and 1.
            n += 1.0f;
            n *= 0.5f;
            noiseArr[j*W+i] = n;
        }
    }

}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
    info.w = W;
    info.h = H;
    info.n_frame = NFRAME;
    // fps = 24/1 = 24
    info.fps_n = 24;
    info.fps_d = 1;
};


void Lab1VideoGenerator::Generate(uint8_t *yuv) {

    for(int i=0 ; i<W*H ; i++) {
        float tmp = noiseArr[i];
//        fprintf(stderr, "%f %d\n", tmp, (int)(tmp * 255)); 
        cudaMemset(yuv+i, (int)(tmp * 255), 1);
    }
    cudaMemset(yuv+W*H, 128, W*H/2);
}

