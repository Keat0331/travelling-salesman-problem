#pragma once
#include "TspSolverCuda.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "device_launch_parameters.h"

struct CityDistance {
    double distance;
    int cityIdx;
};


//----------------------------------------------PRECOMPUTEDISTANCES-----------------------------------------------------------

__global__ void computeDistancesKernel(const City* cities, double* distanceMatrix, int n);

void CudaComputeDistancesKernel(const City* cities, double* distanceMatrix, int n);

//----------------------------------------------CREATECANDIDATELIST-----------------------------------------------------------

__global__ void findTopKClosestCitiesKernel(double* __restrict__ distances, CityDistance* __restrict__ closestCities, int numCities, int k);

void findTopKClosestCities(const double* distances, CityDistance* closestCities, int numCities, int k);

//------------------------------------------twoopt----------------------------------------------------------------

__global__ void twoOptKernel(const double* __restrict__ distanceMatrix, int* __restrict__ tour, int* __restrict__ candidateList, int* __restrict__ positionMap, int numCities, double* globalMinChanges, int* globalIMin, int* globalJMin, int k);

__global__ void reverseSegment(int* __restrict__ tour, int* __restrict__ positionMap, int start, int end);

__global__ void findMinChangeKernel(const double* __restrict__ globalMinChanges, const int* __restrict__ globalIMin, const int* __restrict__ globalJMin, int numCities, double* __restrict__ globalMinChange, int* __restrict__ globalMinI, int* __restrict__ globalMinJ);

void twoOptCuda(const double* h_distanceMatrix, int* h_tour, int* h_candidateList, int numCities, int k);
