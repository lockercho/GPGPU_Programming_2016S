#include "Perlin2D.h"

Perlin2D::Perlin2D() {
     p = new int[size];
     gradient_x = new float[size];
     gradient_y = new float[size];

    for (int i = 0; i < size; ++i ) {
        int other = rand() % (i + 1);
        if (i > other) {
            p[i] = p[other];
        }
        p[other] = i;
        gradient_x[i] = cosf( 2.0f * M_PI * i / size );
        gradient_y[i] = sinf( 2.0f * M_PI * i / size );
    }
}

float f(float t) {
    t = fabsf(t);
    return t >= 1.0f ? 0.0f : 1.0f - ( 3.0f - 2.0f * t ) * t * t;
}
float surflet(float x, float y, float grad_x, float grad_y) {
    return f(x) * f(y) * (grad_x * x + grad_y * y);
}

float Perlin2D::noise(float x, float y) {
    float result = 0.0f;
    int x0 = floorf(x);
    int y0 = floorf(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    for (int grid_y = y0 ; grid_y <= y0 + 1; ++grid_y ) {
        for (int grid_x = x0; grid_x <= x0 + 1; ++grid_x ) {
            int hash = p[( p[ grid_x & mask ] + grid_y ) & mask];
            result += surflet(x - grid_x, y - grid_y, gradient_x[hash], gradient_y[ hash ]);
        }
    }
    return result;
}

float Perlin2D::getFractal(float x, float y, float freq) {
     float sum = 0;
     float amp = amplitude;
     for (int i=0; i<octaves; ++i) {
          float tmp = noise(x * freq, y * freq) * amp;
          sum += tmp;
          freq *= lacunarity;
          amp *= persistence;
     }
     return sum;
}