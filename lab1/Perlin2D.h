#include <cstdlib>
#include <ctime>
#include <cmath>
#include <stdio.h>

class Perlin2D {
     int size = 256;
     int mask = 255;
     int * p;
     float * gradient_x;
     float * gradient_y;
     float amplitude = 1.0;
     float lacunarity = 2.0f;
     float persistence = 0.5f;
     int octaves = 8;
public:
     Perlin2D();
     float noise(float x, float y);
     float getFractal(float x, float y, float freq);
};