//////////////////////////////////////////////////////
//           	     Developed by 		    //
//	   Noskov Roman <noskovrg@gmail.com>	    //
//		     16/may/2016		    //
//////////////////////////////////////////////////////

// Compile: nvcc -lGL -lGLU -lglut -lGLEW ready.cu 
// zoom+: key "+"
// zoom-: key "-"

#include "libs.h"

__device__ double rosenbrockFunction(double x, double y){
	return pow((1.0 - x), 2) + 100 * pow(y - pow(x, 2), 2);
}

__device__ double schwefel(double x, double y){
	return -x * sin(sqrt(abs(x))) - y * sin(sqrt(abs(y)));
}

__device__ double function(double x, double y){
	return rosenbrockFunction(x, y);
}

__global__ void findBestPositionSwarm(double * position, double * out, int n){
	__shared__ double sdata[2 * 1024];
	int tid = threadIdx.x;
	int index = blockIdx.x * blockDim.x * 2 + tid;
	if (index + blockDim.x < n){
		if (function(position[index * 2], position[index * 2 + 1]) < function(position[(index + blockDim.x) * 2], position[(index + blockDim.x) * 2 + 1])){
			sdata[tid * 2] = position[index * 2];
			sdata[tid * 2 + 1] = position[index * 2 + 1];
		}
		else{
			sdata[tid * 2] = position[(index + blockDim.x) * 2];
			sdata[tid * 2 + 1] = position[(index + blockDim.x) * 2 + 1];
		}
	}
	else if (index < n){
		sdata[tid * 2] = position[2 * index];
		sdata[tid * 2 + 1] = position[2 * index + 1]; 
	}
	else{
		sdata[tid * 2] = position[0];
		sdata[tid * 2 + 1] = position[1];
	}
	__syncthreads();
	for (int s = blockDim.x / 2; s > 0; s >>= 1){
		if (tid < s){
			if (function(sdata[(tid + s) * 2], sdata[(tid + s) * 2 + 1]) < function(sdata[tid * 2], sdata[tid * 2 + 1])){
				sdata[tid * 2] = sdata[(tid + s) * 2];
				sdata[tid * 2 + 1] = sdata[(tid + s) * 2 + 1];
			}
		}
		__syncthreads();
	 }
	if (tid == 0){
		out[blockIdx.x] = sdata[0];
		out[blockIdx.x + 1] = sdata[1];
	}
}

__global__ void setBestPositionSwarm(double * newValues, double * bestValues){
	if (function(newValues[0], newValues[1]) < function(bestValues[0], bestValues[1])){
		bestValues[0] = newValues[0];
		bestValues[1] = newValues[1];
	}
}

void updateBestPositionSwarm(double * positionParticleDev, int size){
	int cntBlock = size / (2 * threads1d) + 1;
	double * mem = NULL;
	cudaMalloc((void**)&mem, sizeof(double) * 2 * cntBlock);
	findBestPositionSwarm<<<cntBlock, threads1d>>>(positionParticleDev, mem, size);
	if (cntBlock > 1){
		updateBestPositionSwarm(mem, cntBlock);
	}
	else{
		setBestPositionSwarm<<<1, 1>>>(mem, bestPositionSwarmDev);
	}
	cudaFree(mem);
}

__global__ void sums(double * values, double * out, int n){
	__shared__ double sdata[2 * 1024];
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
	if (i + blockDim.x < n){
			sdata[tid * 2] = values[i * 2] + values[(i + blockDim.x) * 2];
			sdata[tid * 2 + 1] = values[i * 2 + 1] + values[(i + blockDim.x) * 2 + 1];
	}
	else if (i < n){
		sdata[tid * 2] = values[2 * i];
		sdata[tid * 2 + 1] = values[2 * i + 1]; 
	}
	else{
		sdata[tid * 2] = 0;
		sdata[tid * 2 + 1] = 0;
	}
	__syncthreads();
	for (int s = blockDim.x / 2; s > 0; s >>= 1){
		if (tid < s){
			sdata[tid * 2] += sdata[(tid + s) * 2];
			sdata[tid * 2 + 1] += sdata[(tid + s) * 2 + 1];
		}
		__syncthreads();
	 }
	if (tid == 0 ){
		out[blockIdx.x] = sdata[0];
		out[blockIdx.x + 1] = sdata[1];
	}
}

// calculating of function at the pixel (i, j)
__global__ void updateValueMap(double * valueMap) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	double sizePixelX = scaleXDev[0] / windowSize;
	double sizePixelY = scaleYDev[0] / windowSize;
	for(int j = idx; j < windowSize; j += offsetx){
		for (int i = idy; i < windowSize; i += offsety){
			// screen coordinates to world coordinates 
			double x = (j - windowSize / 2) * sizePixelX + centerXDev[0];
			double y = (i - windowSize / 2) * sizePixelY + centerYDev[0];
			valueMap[i * windowSize + j] = function(x, y);
		}
	}
}

__global__ void minKernel(double * values, double * out, int n){
	__shared__ double sdata[1024];
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
	if (i + blockDim.x < n){
		sdata[tid] = min(values[i], values[i + blockDim.x]);
	}
	else if (i < n){
		sdata[tid] = values[i];
	}
	else{
		sdata[tid] = values[0];
	}
	__syncthreads();
	for (int s = blockDim.x / 2; s > 0; s >>= 1){
		if (tid < s){
			sdata[tid] = min(sdata[tid + s], sdata[tid]);
		}
		__syncthreads();
	 }
	if (tid == 0){
		out[blockIdx.x] = sdata[0];
	}
}

__global__ void maxKernel(double * values, double * out, int n){
	__shared__ double sdata[1024];
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
	if (i + blockDim.x < n){
		sdata[tid] = max(values[i], values[i + blockDim.x]);
	}
	else if (i < n){
		sdata[tid] = values[i];
	}
	else{
		sdata[tid] = values[0];
	}
	__syncthreads();
	for (int s = blockDim.x / 2; s > 0; s >>= 1){
		if (tid < s){
			sdata[tid] = max(sdata[tid + s], sdata[tid]);
		}
		__syncthreads();
	 }
	if (tid == 0){
		out[blockIdx.x] = sdata[0];
	}
}

// search the minimum of valueMap with by reduction
// mem - is array of the best threads1d * cntBlock values
void findMin(double * values, int size){
	int cntBlock = size / (2 * threads1d) + 1;
	double * mem = NULL;
	cudaMalloc((void**)&mem, sizeof(double) * cntBlock);
	minKernel<<<cntBlock, threads1d>>>(values, mem, size);
	if (cntBlock > 1){
		findMin(mem, cntBlock);
	}
	else{
		cudaMemcpy(&minf, mem, sizeof(double), cudaMemcpyDeviceToHost);		
		cudaMemcpyToSymbol(minfDev, &minf, sizeof(double));
	}
	cudaFree(mem);
}

// search the maximum of valueMap with by reduction
// mem - is array of the best threads1d * cntBlock values
void findMax(double * values, int size){
	int cntBlock = size / (2 * threads1d) + 1;
	double * mem = NULL;
	cudaMalloc((void**)&mem, sizeof(double) * cntBlock);
	maxKernel<<<cntBlock, threads1d>>>(values, mem, size);
	if (cntBlock > 1){
		findMax(mem, cntBlock);
	}
	else{
		cudaMemcpy(&maxf, mem, sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpyToSymbol(maxfDev, &maxf, sizeof(double));
	}
	cudaFree(mem);
}

void updateValue(){	
	updateValueMap<<<blocks1d, threads1d>>>(valueMapDev);
	findMin(valueMapDev, windowSize * windowSize);
	findMax(valueMapDev, windowSize * windowSize);
}

__global__ void launchParticles(curandState_t * randomStates, long long seed, double * position, double * velocity) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;
	for (int i = idx; i < amountParticle; i += offsetx){
		curand_init(seed, i, 0, &randomStates[i]);
		position[i * 2] = (centerXDev[0] - scaleXDev[0] / 2) + curand_uniform(&randomStates[i]) * scaleXDev[0];
		position[i * 2 + 1] = (centerYDev[0] - scaleYDev[0] / 2) + curand_uniform(&randomStates[i]) * scaleYDev[0];
		velocity[i * 2] = curand_uniform(&randomStates[i]);
		velocity[i * 2 + 1] = curand_uniform(&randomStates[i]);
	}  
}

// updating of the i-th particle's parameters 
__global__ void updateParticles(double * curPosition, double * velocity, double * bestPositionParticle, 
						double * bestPositionSwarm, curandState_t * state, double * repulsiveForce){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;
	for (int i = idx; i < amountParticle; i += offsetx){
		velocity[i * 2] = w * velocity[i * 2] + (iw * curand_uniform(&state[i]) * (bestPositionParticle[i * 2] - curPosition[i * 2]) 
												+ sw * curand_uniform(&state[i]) * (bestPositionSwarm[0] - curPosition[i * 2])) * dt;
		velocity[i * 2 + 1] = w * velocity[i * 2 + 1] + (iw * curand_uniform(&state[i]) * (bestPositionParticle[i * 2 + 1] - curPosition[i * 2 + 1]) 
												+ sw * curand_uniform(&state[i]) * (bestPositionSwarm[1] - curPosition[i * 2 + 1])) * dt;
		if (REPULSIVE_FORCE){
			velocity[i * 2] += dx * repulsiveForce[i * 2] * dt;
			velocity[i * 2 + 1] += dt * repulsiveForce[i * 2 + 1] * dt; 
		}
		curPosition[i * 2] = curPosition[i * 2] + velocity[i * 2];
		curPosition[i * 2 + 1] = curPosition[i * 2 + 1] + velocity[i * 2 + 1]; 
		if (function(curPosition[i * 2], curPosition[i * 2 + 1]) < function(bestPositionParticle[i * 2], bestPositionParticle[i * 2 + 1])){
			bestPositionParticle[i * 2] = curPosition[i * 2];
			bestPositionParticle[i * 2 + 1] = curPosition[i * 2 + 1];
		}
	}
}

__global__ void updateHotMap(uchar4 * pixMap, double * valueMap) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	for(int j = idx; j < windowSize; j += offsetx){
		for (int i = idy; i < windowSize; i += offsety){
			double value = valueMap[i * windowSize + j];
			double normedValue = (value - minfDev[0]) / (maxfDev[0] - minfDev[0]);
			pixMap[i * windowSize + j] = make_uchar4(
				value <= MAX_VALUE_RED_ZONE ? value / MAX_VALUE_RED_ZONE * UCHAR_MAX : 0, (int)(normedValue * UCHAR_MAX), 0, 0);	
		}
	}
}


__global__ void drawParticles(uchar4 * pixMap, double * coords) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;
	double sizePixelX = scaleXDev[0] / windowSize;
	double sizePixelY = scaleYDev[0] / windowSize;
	for(int i = idx; i < amountParticle; i += offsetx){
		// world coordinates to screen coordinates
		int x = (coords[i * 2] - centerXDev[0]) / sizePixelX + windowSize / 2;
		int y = (coords[i * 2 + 1] - centerYDev[0]) / sizePixelY + windowSize / 2;
		if (x >= 0 && x < windowSize && y >= 0 && y < windowSize){
			pixMap[y * windowSize + x] = make_uchar4(0, 255, 0, 0);
			pixMap[y * windowSize + min(windowSize - 1, x + 1)] = make_uchar4(0, 255, 0, 0);
			pixMap[y * windowSize + max(0, x - 1)] = make_uchar4(0, 255, 0, 0);
			pixMap[max((y - 1), 0) * windowSize + x] = make_uchar4(0, 255, 0, 0);
			pixMap[min((y + 1), windowSize - 1) * windowSize + x] = make_uchar4(0, 255, 0, 0);
			pixMap[y * windowSize + min(windowSize - 1, x + 2)] = make_uchar4(0, 255, 0, 0);
			pixMap[y * windowSize + max(0, x - 2)] = make_uchar4(0, 255, 0, 0);
			pixMap[max((y - 2), 0) * windowSize + x] = make_uchar4(0, 255, 0, 0);
			pixMap[min((y + 2), windowSize - 1) * windowSize + x] = make_uchar4(0, 255, 0, 0);
		}
	}
}

// calculating repulsive force for the bid-th particle with by warp-reduction
__global__ void updateRepulsiveForces(double * forces, int n, double * coords){
	int bid = blockIdx.x, tid = threadIdx.x;
	// x, y - Ox & Oy projections of repulsive forces
	__shared__ double x[warp], y[warp];
	if (tid == 0){
		forces[2 * bid] = 0;
		forces[2 * bid + 1] = 0;
	}
	x[tid] = y[tid] = 0;
	__syncthreads();
	double bidX = coords[2 * bid], bidY = coords[2 * bid + 1];
	for (int i = tid; i < n; i += warp){
		if (i != bid){
			x[tid] += (bidX - coords[i * 2]) / (pow(hypot(bidX - coords[i * 2], bidY - coords[i * 2 + 1]), 4) + eps);
			y[tid] += (bidY - coords[i * 2 + 1]) / (pow(hypot(bidX - coords[i * 2], bidY - coords[i * 2 + 1]), 4) + eps);
		}
	}
	__syncthreads();
	for (int s = 16; s > 0; s >>= 1){
		if (tid < s){
			x[tid] = x[tid + s] + x[tid];
			y[tid] = y[tid + s] + y[tid];
		}
		__syncthreads();
	}
	if (tid == 0){
		forces[bid * 2] = x[0];
		forces[bid * 2 + 1] = y[0];
	}

}

// calculating the center of mass and updating the position of window center
// with by reduction
void updateCenter(double * position, int amountPosition){
	int cntBlock = amountPosition / (2 * threads1d) + 1;
	double * newPosition = NULL;
	cudaMalloc((void**)&newPosition, sizeof(double) * dimension * cntBlock);
	sums<<<cntBlock, threads1d>>>(position, newPosition, amountPosition);
	if (cntBlock > 1){
		updateCenter(newPosition, cntBlock);
	}
	else{
		cudaMemcpy(&centerX, &newPosition[0], sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(&centerY, &newPosition[1], sizeof(double), cudaMemcpyDeviceToHost);
		centerX /= amountParticle;
		centerY /= amountParticle;
		cudaMemcpyToSymbol(centerXDev, &centerX, sizeof(double));
		cudaMemcpyToSymbol(centerYDev, &centerY, sizeof(double));
	}
	cudaFree(newPosition);
}

void update() {
	uchar4* pixMapDev;
	size_t size;
	CSC(cudaGraphicsMapResources(1, &res, 0));
	CSC(cudaGraphicsResourceGetMappedPointer((void**) &pixMapDev, &size, res));
	updateHotMap<<<blocks3d, threads3d>>>(pixMapDev, valueMapDev);
	drawParticles<<<blocks1d, threads1d>>>(pixMapDev, positionParticleDev);
	if (REPULSIVE_FORCE){
		updateRepulsiveForces<<<amountParticle, warp>>>(repulsiveForces, amountParticle, positionParticleDev);
	}
	updateParticles<<<blocks1d, threads1d>>>(positionParticleDev, velocityParticleDev, bestPositionParticleDev, 
														bestPositionSwarmDev, randomStates, repulsiveForces);
	updateCenter(positionParticleDev, amountParticle);
	updateValue();
	updateBestPositionSwarm(bestPositionParticleDev, amountParticle);
	CSC(cudaDeviceSynchronize());
	CSC(cudaGraphicsUnmapResources(1, &res, 0));
	glutPostRedisplay();
}

void display() {
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glDrawPixels(windowSize, windowSize, GL_RGBA, GL_UNSIGNED_BYTE, 0);	
	glutSwapBuffers();
}

void initSwarm(){
	cudaMemcpyToSymbol(scaleXDev, &scaleX, sizeof(double));
	cudaMemcpyToSymbol(scaleYDev, &scaleY, sizeof(double));
	cudaMemcpyToSymbol(centerXDev, &centerX, sizeof(double));
	cudaMemcpyToSymbol(centerYDev, &centerY, sizeof(double));
	cudaMalloc((void**)&bestPositionSwarmDev, dimension * sizeof(double));	
	cudaMalloc((void**)&randomStates, amountParticle * sizeof(curandState_t));
	cudaMalloc((void**)&positionParticleDev, dimension * amountParticle * sizeof(double));
	cudaMalloc((void**)&velocityParticleDev, dimension * amountParticle * sizeof(double));	
	launchParticles<<<blocks1d, threads1d>>>(randomStates, time(NULL), positionParticleDev, velocityParticleDev);
	cudaMalloc((void**)&repulsiveForces, dimension * amountParticle * sizeof(double));	
	cudaMalloc((void**)&bestPositionParticleDev, dimension * amountParticle * sizeof(double));
	cudaMemcpy(bestPositionParticleDev, positionParticleDev, sizeof(double) * dimension * amountParticle, cudaMemcpyDeviceToDevice);	
	cudaMemcpy(bestPositionSwarmDev, bestPositionParticleDev, sizeof(double) * dimension, cudaMemcpyDeviceToDevice);
	updateBestPositionSwarm(bestPositionParticleDev, amountParticle);
	updateCenter(positionParticleDev, amountParticle);
	CSC(cudaMalloc((void**)&valueMapDev, sizeof(double) * windowSize * windowSize));
	updateValue();
}

void killSwarm(){
	CSC(cudaFree(positionParticleDev));
	CSC(cudaFree(velocityParticleDev));
	CSC(cudaFree(bestPositionParticleDev));
	CSC(cudaFree(randomStates));
	CSC(cudaFree(repulsiveForces));
	CSC(cudaFree(valueMapDev));
}

void KeyboardEvent(unsigned char key, int x, int y){   
	if (key == '-'){
		scaleX = min(scaleX + 0.25, 50.0);
		scaleY = min(scaleY + 0.25, 50.0);
		cudaMemcpyToSymbol(scaleXDev, &scaleX, sizeof(double));
		cudaMemcpyToSymbol(scaleYDev, &scaleY, sizeof(double));
	}
	else if (key == '+'){
		scaleX = max(scaleX - 0.25, 0.5);
		scaleY = max(scaleY - 0.25, 0.5);
		cudaMemcpyToSymbol(scaleXDev, &scaleX, sizeof(double));
		cudaMemcpyToSymbol(scaleYDev, &scaleY, sizeof(double));
	}
}

int main(int argc, char** argv){
	initSwarm();
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(windowSize, windowSize);
	glutCreateWindow("Hot map");
	glutIdleFunc(update);
	glutDisplayFunc(display);
	glutKeyboardFunc(KeyboardEvent);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble) windowSize, 0.0, (GLdouble) windowSize);
	glewInit();
	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, vbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, windowSize * windowSize * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);
	CSC(cudaGraphicsGLRegisterBuffer(&res, vbo, cudaGraphicsMapFlagsWriteDiscard));
	glutMainLoop();
	CSC(cudaGraphicsUnregisterResource(res));
	glBindBuffer(1, vbo);
	glDeleteBuffers(1, &vbo);
	killSwarm();
	return 0;
}
