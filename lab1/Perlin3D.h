#ifndef _PERLIN3D_H_
#define _PERLIN3D_H_


class Perlin3D {
     float amplitude = 1.0;
     float lacunarity = 2.0f;
     float persistence = 0.5f;
     int octaves = 8;
     int * permutation;
     float * gradient_x;
     float * gradient_y;
     float * gradient_z;
public:
     Perlin3D();
     ~Perlin3D();

     float noise(float x, float y, float z);
     float getFractal(float x, float y, float z, float freq);
};

#endif
