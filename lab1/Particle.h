#ifndef _PARTICAL_H_
#define _PARTICAL_H_

#include "const.h"

class Particle {
public:
	Particle(double x, double y, double w) {
		posX = x, posY = y, weight = w;
		fx = fy = sx = sy = 0;
	}
	double posX;
	double posY;
	double rarius = 3;
	double weight;
	double fx;
	double fy;
	double sx;
	double sy;
	int iter = -1;
	void setF(double _fx, double _fy, int _iter) {
		if(_iter > iter) {
			iter = _iter;
			fx = 0, fy = 0;
		}
		fx += _fx;
		fy += _fy;
	}

	void clearF() {
		fx = 0;
		fy = 0;
	}

	void setS(double _sx, double _sy) {
		sx = _sx;
		sy = _sy;
	}
	void move(double t) {
		posX += sx * t + fx / weight * t * t / 2;
		posY += sy * t + fy / weight * t * t / 2;
        // maintain current speed
		sx += fx / weight * t;
		sy += fy / weight * t;
	}
};

#endif
