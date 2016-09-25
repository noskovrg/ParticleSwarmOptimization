#ifndef LIBS_H
#define LIBS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <cassert>

#define REPULSIVE_FORCE 1
#define eps 0
#define dimension 2
#define warp 32
#define MAX_VALUE_RED_ZONE 10.0
#define CSC(call) {							\
    cudaError err = call;						\
    if(err != cudaSuccess) {						\
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",	\
            __FILE__, __LINE__, cudaGetErrorString(err));		\
        exit(1);							\
    }									\
} while (0)

const int windowSize = 768; 
const int amountParticle = 1024;
const int blocks1d = 1024, threads1d = 1024;

dim3 blocks3d(32, 32, 1), threads3d(32, 32, 1);
__constant__ double maxfDev[1], minfDev[1], centerXDev[1], centerYDev[1], scaleXDev[1], scaleYDev[1];
double centerX = 5, centerY = 5, scaleX = 10, scaleY = 10, maxf = 14416.0, minf = 0.0;
double *positionParticleDev = NULL, *velocityParticleDev = NULL, *bestPositionParticleDev = NULL, 
		 *repulsiveForces = NULL, *bestPositionSwarmDev = NULL, *valueMapDev = NULL;
__device__ double iw =  0.01, sw = 0.1, w = 0.01, dx = 0.01, dy = 0.1, dt = 0.01;
curandState_t * randomStates = NULL;
struct cudaGraphicsResource *res = NULL;

using namespace std;

#endif