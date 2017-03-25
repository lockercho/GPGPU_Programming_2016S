#include "lab1.h"


struct Lab1VideoGenerator::Impl {
	int t = 0;
};

void Lab1VideoGenerator::generateNoise(float * noiseArr, float freq) {
	int noise_width = W * 2;
	int noise_height = H * 2;

	noiseArr = new float[noise_width * noise_height];
	// Initialize noise generator
	noiseMaker->setBaseFrequency(freq);

	// Generate a noise value for each pixel
	float invWidth = 1.0f / float(noise_width);
	float invHeight = 1.0f / float(noise_height);
	float noise;
	float min = 0.0f;
	float max = 0.0f;

	for (int x=0; x<noise_width; ++x) for (int y=0; y<noise_height; ++y) {
		
		noise = noiseMaker->noise(float(x)*invWidth, float(y)*invHeight, 0.72);

		noiseArr[y*noise_width + x] = noise;

		// Keep track of minimum and maximum noise values
		if (noise < min) min = noise;
		if (noise > max) max = noise;
	}

	// Convert noise values to pixel colour values.
	float temp = 1.0f / (max - min);

	for (int x=0; x<noise_width; ++x) for (int y=0; y<noise_height; ++y) {
		
		// "Stretch" the gaussian distribution of noise values to better fill -1 to 1 range.
		noise = noiseArr[y*noise_width + x];
		noise = -1.0f + 2.0f*(noise - min)*temp;
		// Remap to RGB friendly colour values in range between 0 and 1.
		noise += 1.0f;
		noise *= 0.5f;
		noiseArr[y*noise_width + x] = noise;
	}	
}


float Lab1VideoGenerator::getNoise(float * noiseArr, int x, int y) {
	int noise_width = W * 2;
	int noise_height = H * 2;
	return noiseArr[(y + H /2 ) * noise_width + (x + W / 2 )];
}

void Lab1VideoGenerator::setRotMatrix(int degree) {
	rotMat[0][0] = cos(degree * M_PI / 180);
	rotMat[0][1] = -sin(degree * M_PI / 180);
	rotMat[1][0] = sin(degree * M_PI / 180);
	rotMat[1][1] = cos(degree * M_PI / 180);
}

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
    noiseMaker = new FractalNoise();
    // generateNoise(loose_noise, 1);
    generateNoise(dense_noise, 8);
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
	// define rotation per second
	setRotMatrix(impl->t);
	for(int i=0 ; i<W*H ; i++) {
		cudaMemset(yuv+i, getNoise(dense_noise, i % W, i / W) * 255, 1);
	}
	cudaMemset(yuv+W*H, 128, W*H/2);
}

