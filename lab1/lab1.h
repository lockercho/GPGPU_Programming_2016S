#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include "Perlin2D.h"
#include "Particle.h"
#include "const.h"
using namespace std;
using std::unique_ptr;

struct Lab1VideoInfo {
	unsigned w, h, n_frame;
	unsigned fps_n, fps_d;
};

class Lab1VideoGenerator {
	struct Impl;
	unique_ptr<Impl> impl;
	Perlin2D * noiseMaker;
	int W = WIDTH;
	int H = HEIGHT;
    int seconds = SECONDS;
    int fps = FPS;
	int NFRAME = fps * seconds;
	float rotMat[2][2];
	float * loose_noise;
	float * dense_noise;
	vector<Particle> particles;
public:
	Lab1VideoGenerator();
	~Lab1VideoGenerator();
	void get_info(Lab1VideoInfo &info);
	void Generate(uint8_t *yuv);
	float getNoise(float * noiseArr, int x, int y); 
	void setRotMatrix(int degree);
	void generateNoise(float * noiseArr, float freq);
    void rotate(int &x, int &y);
    void rotateAndFade(uint8_t *yuv);
    void gravitySimulation(uint8_t * yuv);
};
