#include "Perlin3D.h"

#include <cstdlib>
#include <ctime>
#include <cmath>

Perlin3D::Perlin3D() {
    // srand(time(NULL));
    srand(100);

    permutation = new int[256];
    gradient_x = new float[256];
    gradient_y = new float[256];
    gradient_z = new float[256];

    for (int i=0; i<256; ++i) {
        permutation[i] = i;
        gradient_x[i] = (float(rand()) / (RAND_MAX/2)) - 1.0f;
        gradient_y[i] = (float(rand()) / (RAND_MAX/2)) - 1.0f;
        gradient_z[i] = (float(rand()) / (RAND_MAX/2)) - 1.0f;
    }

    int j=0;
    int swp=0;
    for (int i=0; i<256; i++) {
        j = rand() & 255;

        swp = permutation[i];
        permutation[i] = permutation[j];
        permutation[j] = swp;
    }
}

Perlin3D::~Perlin3D()
{
    delete permutation;
    delete gradient_x;
    delete gradient_y;
    delete gradient_z;
}


float Perlin3D::noise(float sample_x, float sample_y, float sample_z)
{
    // Unit cube vertex coordinates surrounding the sample point
    int x0 = int(floorf(sample_x));
    int x1 = x0 + 1;
    int y0 = int(floorf(sample_y));
    int y1 = y0 + 1;
    int z0 = int(floorf(sample_z));
    int z1 = z0 + 1;

    // Determine sample point position within unit cube
    float px0 = sample_x - float(x0);
    float px1 = px0 - 1.0f;
    float py0 = sample_y - float(y0);
    float py1 = py0 - 1.0f;
    float pz0 = sample_z - float(z0);
    float pz1 = pz0 - 1.0f;

    // Compute dot product between gradient and sample position vector
    int gIndex = permutation[(x0 + permutation[(y0 + permutation[z0 & 255]) & 255]) & 255];
    float d0 = gradient_x[gIndex]*px0 + gradient_y[gIndex]*py0 + gradient_z[gIndex]*pz0;
    gIndex = permutation[(x1 + permutation[(y0 + permutation[z0 & 255]) & 255]) & 255];
    float d1 = gradient_x[gIndex]*px1 + gradient_y[gIndex]*py0 + gradient_z[gIndex]*pz0;
    
    gIndex = permutation[(x0 + permutation[(y1 + permutation[z0 & 255]) & 255]) & 255];
    float d2 = gradient_x[gIndex]*px0 + gradient_y[gIndex]*py1 + gradient_z[gIndex]*pz0;
    gIndex = permutation[(x1 + permutation[(y1 + permutation[z0 & 255]) & 255]) & 255];
    float d3 = gradient_x[gIndex]*px1 + gradient_y[gIndex]*py1 + gradient_z[gIndex]*pz0;
    
    gIndex = permutation[(x0 + permutation[(y0 + permutation[z1 & 255]) & 255]) & 255];
    float d4 = gradient_x[gIndex]*px0 + gradient_y[gIndex]*py0 + gradient_z[gIndex]*pz1;
    gIndex = permutation[(x1 + permutation[(y0 + permutation[z1 & 255]) & 255]) & 255];
    float d5 = gradient_x[gIndex]*px1 + gradient_y[gIndex]*py0 + gradient_z[gIndex]*pz1;

    gIndex = permutation[(x0 + permutation[(y1 + permutation[z1 & 255]) & 255]) & 255];
    float d6 = gradient_x[gIndex]*px0 + gradient_y[gIndex]*py1 + gradient_z[gIndex]*pz1;
    gIndex = permutation[(x1 + permutation[(y1 + permutation[z1 & 255]) & 255]) & 255];
    float d7 = gradient_x[gIndex]*px1 + gradient_y[gIndex]*py1 + gradient_z[gIndex]*pz1;

    // Interpolate dot product values at sample point using polynomial interpolation 6x^5 - 15x^4 + 10x^3
    float wx = ((6*px0 - 15)*px0 + 10)*px0*px0*px0;
    float wy = ((6*py0 - 15)*py0 + 10)*py0*py0*py0;
    float wz = ((6*pz0 - 15)*pz0 + 10)*pz0*pz0*pz0;

    float xa = d0 + wx*(d1 - d0);
    float xb = d2 + wx*(d3 - d2);
    float xc = d4 + wx*(d5 - d4);
    float xd = d6 + wx*(d7 - d6);
    float ya = xa + wy*(xb - xa);
    float yb = xc + wy*(xd - xc);
    float value = ya + wz*(yb - ya);

    return value;
}

float Perlin3D::getFractal(float x, float y, float z, float freq) {
    float sum = 0;
    float amp = amplitude;
    for (int i=0; i<octaves; ++i) {
        float tmp = noise(x * freq, y * freq, z* freq) * amp;
        sum += tmp;
        freq *= lacunarity;
        amp *= persistence;
    }
    return sum;
}
