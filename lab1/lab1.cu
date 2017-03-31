#include "lab1.h"
#include "const.h"
#include "Particle.h"
#include "Perlin3D.h"

bool isCollision(Particle * p1, Particle * p2) {
    double dx = p2->posX - p1->posX;
    double dy = p2->posY - p1->posY;
    double r2 = dx*dx + dy*dy;
    double r = sqrt(r2);
    //    fprintf(stderr, "r: %f\n", r);
    return (r <= p1-> radius || r <= p2-> radius);
}

void getF(Particle * p1, Particle * p2, double &fx, double &fy) {
    double dx = p2->posX - p1->posX;
    double dy = p2->posY - p1->posY;
    double r2 = dx*dx + dy*dy;
    double r = sqrt(r2);
    double F = G * p1->weight * p2->weight / r2;
    fx = F * dx / r;
    fy = F * dy / r;
}
struct Lab1VideoGenerator::Impl {
    int t = 0;
};

void Lab1VideoGenerator::generateNoise(float * noiseArr, float freq) {
    
       int noise_width = W * 2;
       int noise_height = H * 2;

    // Generate a noise value for each pixel
    float invWidth = 1.0f / float(noise_width);
    float invHeight = 1.0f / float(noise_height);
    float invZ = 1.0f / float(NFRAME);
    float noise;
    float min = 0.0f;
    float max = 0.0f;

    for (int x=0; x<noise_width; ++x) {
        for (int y=0; y<noise_height; ++y) {
            for(int z=0; z<NFRAME; ++z) {
                noise = noiseMaker->getFractal(float(x)*invWidth, float(y)*invHeight, float(z)*invZ, freq);

                noiseArr[z*noise_width*noise_height+ y*noise_width + x] = noise;

                // Keep track of minimum and maximum noise values
                if (noise < min) min = noise;
                if (noise > max) max = noise;
            }
        }
    }

    // Convert noise values to pixel colour values.
    float temp = 1.0f / (max - min);

    for (int x=0; x<noise_width; ++x) {
        for (int y=0; y<noise_height; ++y) {
            for(int z=0; z<NFRAME; ++z) {
                // "Stretch" the gaussian distribution of noise values to better fill -1 to 1 range.
                noise = noiseArr[z*noise_width*noise_height + y*noise_width + x];
                noise = -1.0f + 2.0f*(noise - min)*temp;
                // Remap to RGB friendly colour values in range between 0 and 1.
                noise += 1.0f;
                noise *= 0.5f;
                noiseArr[z*noise_width*noise_height + y*noise_width + x] = noise;
            }
        }
    }
}


float Lab1VideoGenerator::getNoise(float * noiseArr, int x, int y, int z) {
    int noise_width = W * 2;
    int noise_height = H * 2;
    return noiseArr[z * noise_width * noise_height + (y + H /2 ) * noise_width + (x + W / 2 )];
}

void Lab1VideoGenerator::setRotMatrix(int degree) {
    rotMat[0][0] = cos(degree * M_PI / 180);
    rotMat[0][1] = -sin(degree * M_PI / 180);
    rotMat[1][0] = sin(degree * M_PI / 180);
    rotMat[1][1] = cos(degree * M_PI / 180);
}

void Lab1VideoGenerator::rotate(int &x, int &y) {
    // normalize it
    float nx = float(x) / W - 0.5;
    float ny = float(y) / H - 0.5;
    x = int((nx * rotMat[0][0] + ny * rotMat[0][1] + 0.5) * W);
    y = int((nx * rotMat[1][0] + ny * rotMat[1][1] + 0.5) * H);
}

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
    noiseMaker = new Perlin3D();
    loose_noise = new float[W*H*2*2];
    dense_noise = new float[W*H*2*2];
       generateNoise(loose_noise, 1);
       generateNoise(dense_noise, 8);

    // init gravity
    // int w_size = W_SIZE;
    // int h_size = H_SIZE;
    // for(int i=1 ; i<=w_size ; i++) {
    //     for(int j=1; j<=h_size; j++) {
    //         double x = (double) W / w_size * i;
    //         double y = (double) H / h_size * j;
    //         double w = pow(2, (w_size / 2 - abs(w_size/2-i) + h_size / 2 - abs(h_size/2 -j)) / 3 );
    //         fprintf(stderr, "xyw: %f %f %f\n", x, y, w);
    //         particles.push_back(Particle(x, y, w));
    //     }
    // }
    //exit(0);
    /*
       particles.push_back(Particle(0.0, 0.0, 10));
       particles.push_back(Particle(1.0, 0.0, 10));
       particles.push_back(Particle(1.0, 1.0, 10));
       particles.push_back(Particle(0.0, 1.0, 10));
     */
}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
    info.w = W;
    info.h = H;
    info.n_frame = NFRAME;
    // fps = 24/1 = 24
    info.fps_n = fps;
    info.fps_d = 1;
};

void Lab1VideoGenerator::Generate(uint8_t *yuv) {
    //intoTheFog(yuv);
    // gravitySimulation(yuv);
    rotateAndFade(yuv);
}

void Lab1VideoGenerator::intoTheFog(uint8_t *yuv) {
    // rotate and fade transition
    int loop = fps * 2;  
    float w = float(impl->t % loop) / loop;
    w = w * w;
    int direction = impl->t / loop % 2;
    setRotMatrix(impl->t * 24 / fps);
    float z = float(impl-> t % W) / W;
    for(int i=0 ; i<W*H ; i++) {
        int ix = i % W;
        int iy = i / W;
        rotate(ix, iy); 
        float x = float(ix) / W;
        float y = float(iy) / H;
        float noise = noiseMaker->getFractal(x, y, z, freq);
        noise += 1.0f;
        noise *= 0.5f;
        float R = (1.0 - noise) * 255;
        float B = (1.0 - noise) * 255 + 100;
        float CG = (1.0 - noise) * 255 + 30;
        if(CG > 255) CG = 255;
        if(B > 255) B = 255;
/*
        int r = int((1.0-p) * r1 + p * r2 + 0.5)
        int g = int((1.0-p) * g1 + p * g2 + 0.5)
        int b = int((1.0-p) * b1 + p * b2 + 0.5)
*/
        //fprintf(stderr, "xyz: %f %f %f %f %f\n", x, y, z, freq, noise);
        int Y = 0.299 * R + 0.587 * CG + 0.114 * B;
        int U = - 0.169 * R -  0.331 * CG + 0.500 * B + 128;
        int V = 0.500 * R - 0.419 * CG - 0.081 * B + 128;
        cudaMemset(yuv+i, Y, 1);
        if(ix % 2 == 0 && iy %2 == 0) {
            ix /= 2;
            iy /= 2;
            int index = iy * W / 2 + ix;
            cudaMemset(yuv+W*H+index, U, 1);
            cudaMemset(yuv+int(W*H *1.25)+ index, V, 1);
        }
    }
    impl->t+=5;
}

void Lab1VideoGenerator::gravitySimulation(uint8_t * yuv) {
    cudaMemset(yuv, 0, W*H);

    for(int i=0; i< particles.size() ; i++) {
        for(int j=0 ;j< particles.size() ; j++) {
            if(i == j) continue;

            // check collision
            if(isCollision(&particles[i], &particles[j])) {
                fprintf(stderr, "collision\n");
                particles[i].clearF();
                // elastic collision
                Particle p1 = particles[i], p2 = particles[j];
                double sx = (p1.sx * (p1.weight - p2.weight) + 2 * p2.weight * p2.sx ) 
                    / (p1.weight + p2.weight); 
                double sy = (p1.sy * (p1.weight - p2.weight) + 2 * p2.weight * p2.sy ) 
                    / (p1.weight + p2.weight); 
                particles[i].setS(sx, sy);
            } else {
                // check gravity
                //fprintf(stderr, "Not collision\n");
                double gfx = 0, gfy = 0;
                getF(&particles[i], &particles[j], gfx, gfy);
                particles[i].setF(gfx, gfy, impl->t);
            }
        }
    }

    for(int i=0; i< particles.size() ; i++) {
        particles[i].move(1.0 / fps);
        int x = int(particles[i].posX);
        int y = int(particles[i].posY);
        // if out of range, continue
        if(x<0 || x>W || y<0 || y>H) {
            //continue;
        }
        int index = y * W + x;
        /*
           fprintf(stderr, "Pos: %f %f\n", particles[i].posX, particles[i].posY);
           fprintf(stderr, "Real Pos: %d %d %d\n\n", x, y, index);
         */
        cudaMemset(yuv + index, 255, 1);
    }

    cudaMemset(yuv+W*H, 128, W*H/2);
    impl->t++;
}


void Lab1VideoGenerator::rotateAndFade(uint8_t *yuv) {
    // rotate and fade transition
    int loop = fps;
    float w = float(impl->t % loop) / loop;
    w = w * w;
    int direction = impl->t / loop % 2;
    setRotMatrix(impl->t * 24 / fps);
    for(int i=0 ; i<W*H ; i++) {
        int ix, iy;
        int x = ix = i % W;
        int y = iy = i / W;
        rotate(x, y); 
        float n1 = getNoise(loose_noise, x, y, impl->t);
        float n2 = getNoise(dense_noise, x, y, impl->t);
        float color;
        if(direction == 0)
            color = (1.0 - w) * n1 + w * n2;
        else 
            color = w * n1 + (1.0 - w) * n2;

        float R = (1.0 - color) * 155;
        float B = (1.0 - color) * 155 + 100;
        float CG = (1.0 - color) * 155 + 30;
        if(CG > 255) CG = 255;
        if(B > 255) B = 255;

        //fprintf(stderr, "xyz: %f %f %f %f %f\n", x, y, z, freq, noise);
        int Y = 0.299 * R + 0.587 * CG + 0.114 * B;
        int U = - 0.169 * R -  0.331 * CG + 0.500 * B + 128;
        int V = 0.500 * R - 0.419 * CG - 0.081 * B + 128;

        cudaMemset(yuv+i, Y, 1);
        if(ix % 2 == 0 && iy %2 == 0) {
            ix /= 2;
            iy /= 2;
            int index = iy * W / 2 + ix;
            cudaMemset(yuv+W*H+index, U, 1);
            cudaMemset(yuv+int(W*H *1.25)+ index, V, 1);
        }
        // cudaMemset(yuv+i, color * 255, 1);
    }
    // cudaMemset(yuv+W*H, 128, W*H/2);
    impl->t++;
}

