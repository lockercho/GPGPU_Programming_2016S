// F = GMm / (r*r)
#include <stdio.h>
class Particle {
public:
	Particle(double x, double y, double w) {
		posX = x, posY = y, weight = w;
		fx = fy = ax = ay = 0;
	}
	double posX;
	double posY;
	double weight;
	double fx;
	double fy;
	double ax;
	double ay;
	int iter = -1;
	void setF(double _fx, double _fy, int _iter) {
		if(_iter > iter) {
			iter = _iter;
			fx = 0, fy = 0;
		}
		fx += _fx;
		fy += _fy;
	}
	void move(double t) {
		posX += ax * t + fx / weight * t * t / 2;
		posY += ay * t + fy / weight * t * t / 2;
		ax += fx / weight * t;
		ay += fy / weight * t;
	}
};