#include "CudaKernel.cuh"


//----------------------------------------------PRECOMPUTEDISTANCES-----------------------------------------------------------

__global__ void computeDistancesKernel(const City* cities, double* distanceMatrix, int n) {
    // Calculate the row (i) and column (j) indices based on thread and block IDs
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread indices are within bounds
    if (i < n && j < n) {
        if (i != j) {
            // Calculate the Euclidean distance between city i and city j
            double dx = cities[i].x - cities[j].x;
            double dy = cities[i].y - cities[j].y;
            distanceMatrix[i * n + j] = sqrt(dx * dx + dy * dy);
        }
        else {
            // Distance from a city to itself is 0
            distanceMatrix[i * n + j] = 0.0;
        }
    }
}

void CudaComputeDistancesKernel(const City* cities, double* distanceMatrix, int n) {
    City* d_cities;            // Pointer to device memory for cities
    double* d_distanceMatrix;  // Pointer to device memory for the distance matrix

    // Calculate the size of memory needed for cities and distance matrix
    size_t citySize = n * sizeof(City);
    size_t distanceMatrixSize = n * n * sizeof(double);

    // Allocate device memory for cities and distance matrix
    cudaMalloc(&d_cities, citySize);
    cudaMalloc(&d_distanceMatrix, distanceMatrixSize);

    // Copy the cities data from host to device memory
    cudaMemcpy(d_cities, cities, citySize, cudaMemcpyHostToDevice);

    // Define block size and grid size for the kernel launch
    dim3 blockSize(16, 16);  // Block of 16x16 threads
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x,
        (n + blockSize.y - 1) / blockSize.y);  // Grid size to cover all elements

    // Launch the CUDA kernel to compute the distance matrix
    computeDistancesKernel << <gridSize, blockSize >> > (d_cities, d_distanceMatrix, n);

    // Copy the computed distance matrix from device back to host memory
    cudaMemcpy(distanceMatrix, d_distanceMatrix, distanceMatrixSize, cudaMemcpyDeviceToHost);

    // Free the allocated device memory
    cudaFree(d_cities);
    cudaFree(d_distanceMatrix);
}

//----------------------------------------------CREATECANDIDATELIST-----------------------------------------------------------

__device__ void updateClosestCities(CityDistance* closestCities, double dist, int cityIdx, int k) {
    // Determine the position where the new distance should be inserted
    int pos = -1; // Initialize position to -1, indicating no insertion yet
    for (int i = 0; i < k; ++i) {
        // Check if slot is empty or new distance is smaller
        if (closestCities[i].cityIdx == -1 || dist < closestCities[i].distance) {
            pos = i; // Record position for insertion
            break;
        }
    }

    // If a valid position was found, insert the city and shift other cities
    if (pos != -1) {
        // Shift existing cities to make room for the new city
        for (int i = k - 1; i > pos; --i) {
            closestCities[i] = closestCities[i - 1];
        }
        // Insert new city at the identified position
        closestCities[pos].distance = dist;
        closestCities[pos].cityIdx = cityIdx;
    }
}

__global__ void findTopKClosestCitiesKernel(double* __restrict__ distances, CityDistance* __restrict__ closestCities, int numCities, int k) {
    int cityIdx = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the global index for the city

    if (cityIdx < numCities) { // Check if index is within bounds
        double* cityDistances = distances + cityIdx * numCities; // Pointer to the distance array for the current city
        CityDistance* cityClosestCities = closestCities + cityIdx * k; // Pointer to the closestCities for the current city

        // Initialize closestCities with maximum distance
        for (int i = 0; i < k; ++i) {
            cityClosestCities[i].cityIdx = -1; // Set city index to -1 (indicating empty slot)
            cityClosestCities[i].distance = 1e9; // Set distance to a large value
        }

        // Compute distances to all other cities
        for (int j = 0; j < numCities; ++j) {
            if (cityIdx != j) { // Avoid comparing the city to itself
                double dist = cityDistances[j]; // Get distance to city j
                updateClosestCities(cityClosestCities, dist, j, k); // Update closest cities with new distance
            }
        }
    }
}

void findTopKClosestCities(const double* distances, CityDistance* closestCities, int numCities, int k) {
    const int threadsPerBlock = 256; // Number of threads per block
    const int blocksPerGrid = (numCities + threadsPerBlock - 1) / threadsPerBlock; // Number of blocks required

    // Allocate device memory for distances and closestCities
    double* d_distances;
    CityDistance* d_closestCities;

    cudaMalloc(&d_distances, numCities * numCities * sizeof(double)); // Allocate memory for distances on device
    cudaMalloc(&d_closestCities, numCities * k * sizeof(CityDistance)); // Allocate memory for closestCities on device

    // Copy data from host to device
    cudaMemcpy(d_distances, distances, numCities * numCities * sizeof(double), cudaMemcpyHostToDevice); // Copy distances to device

    // Launch the kernel
    findTopKClosestCitiesKernel << <blocksPerGrid, threadsPerBlock >> > (d_distances, d_closestCities, numCities, k);

    // Check for any errors during kernel execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl; // Print error message
        cudaFree(d_distances); // Free device memory
        cudaFree(d_closestCities); // Free device memory
        return;
    }

    // Copy results from device to host
    cudaMemcpy(closestCities, d_closestCities, numCities * k * sizeof(CityDistance), cudaMemcpyDeviceToHost); // Copy closestCities to host

    // Free device memory
    cudaFree(d_distances); // Free device memory
    cudaFree(d_closestCities); // Free device memory
}

//--------------------------------------twoopt-------------------------------------------------------------

// CUDA kernel for performing 2-opt optimization on a TSP tour
__global__ void twoOptKernel(const double* __restrict__ distanceMatrix, 
    int* __restrict__ tour, 
    int* __restrict__ candidateList,
    int* __restrict__ positionMap, 
    int numCities, 
    double* globalMinChanges, 
    int* globalIMin, 
    int* globalJMin, 
    int k) {

    // Calculate the global index of the current thread (i.e., which city it's processing)
    int cityIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread is within bounds (i.e., processing a valid city)
    if (cityIdx < numCities) {
        int a1 = cityIdx;  // The current city (city A1)
        int a2 = tour[(positionMap[a1] + 1) % numCities];  // The next city in the tour (city A2)

        // Fetch the distance between A1 and A2 from the precomputed distance matrix
        double ab = distanceMatrix[a1 * numCities + a2];

        // Get the list of candidate cities to consider for swapping
        const int* candidates = candidateList + cityIdx * k;

        double bestChange = 0.0;  // Initialize the best improvement found to zero (no improvement)
        int bestI = -1, bestJ = -1;  // Initialize the best positions for swapping

        // Loop through each candidate city for potential 2-opt swap
        for (int j = 0; j < k; ++j) {
            int b1 = candidates[j];  // Candidate city B1

            // If the candidate is the first one in the list, skip the rest (no improvement possible)
            if (a2 == candidates[0]) break;

            int b2 = tour[(positionMap[b1] + 1) % numCities];  // The next city after B1 in the tour

            // Ensure we don't swap a city with itself or create a disallowed connection
            if (a2 == b1 || b2 == a1) continue;

            // Calculate the distances for the new potential connections after the swap
            double cd = distanceMatrix[b1 * numCities + b2];  // Distance between B1 and B2
            double ac = distanceMatrix[a1 * numCities + b1];  // Distance between A1 and B1
            double bd = distanceMatrix[a2 * numCities + b2];  // Distance between A2 and B2

            // Calculate the change in tour length if the swap is performed
            double change = (ac + bd) - (ab + cd);

            // If the swap reduces the total distance, update the best change and swap positions
            if (change < bestChange) {
                bestChange = change;
                bestI = positionMap[a1];  // Store the position of A1 in the tour
                bestJ = positionMap[b1];  // Store the position of B1 in the tour
            }
        }

        // Store the best found improvement and corresponding indices in global memory
        globalMinChanges[cityIdx] = bestChange;
        globalIMin[cityIdx] = bestI;
        globalJMin[cityIdx] = bestJ;
    }
}


// CUDA kernel for reversing a segment of the tour
__global__ void reverseSegment(int* __restrict__ tour, int* __restrict__ positionMap, int start, int end) {
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int length = end - start;

    // Ensure the thread index is within the bounds of the segment to be reversed
    if (idx < (length + 1) / 2) {
        // Swap the elements in the tour between start and end
        int temp = tour[start + 1 + idx];
        tour[start + 1 + idx] = tour[end - idx];
        tour[end - idx] = temp;

        // Update the positionMap to reflect the new positions of the swapped elements
        positionMap[tour[start + 1 + idx]] = start + 1 + idx;
        positionMap[tour[end - idx]] = end - idx;
    }
}

// Helper function for performing atomic minimum operations on double precision values
__device__ void atomicMin_double(double* address, double val, int* minI, int newMinI, int* minJ, int newMinJ) {
    // Use unsigned long long int for atomic operations due to limitations with double precision
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    // Perform the atomic minimum operation
    do {
        assumed = old;
        // Check if the current value is less than or equal to the new value
        if (__longlong_as_double(assumed) <= val) break;
        // Atomically update the address with the new minimum value if the current value is greater
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);

    // Update the indices if the new minimum value is indeed lower
    if (__longlong_as_double(assumed) > val) {
        *minI = newMinI;
        *minJ = newMinJ;
    }
}


// CUDA kernel to find the minimum change from the globalMinChanges array
__global__ void findMinChangeKernel(const double* __restrict__ globalMinChanges,
    const int* __restrict__ globalIMin,
    const int* __restrict__ globalJMin,
    int numCities,
    double* __restrict__ globalMinChange,
    int* __restrict__ globalMinI,
    int* __restrict__ globalMinJ) {

    // Shared memory for storing minimum values and their indices
    extern __shared__ double sdata[];
    int* sIMin = (int*)&sdata[blockDim.x];
    int* sJMin = (int*)&sIMin[blockDim.x];

    // Thread index and global index calculation
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Initialize shared memory
    if (i < numCities) {
        sdata[tid] = globalMinChanges[i];
        sIMin[tid] = globalIMin[i];
        sJMin[tid] = globalJMin[i];
    }
    else {
        sdata[tid] = DBL_MAX;  // Initialize with maximum double value
        sIMin[tid] = -1;
        sJMin[tid] = -1;
    }
    __syncthreads();

    // Parallel reduction within the block to find the minimum value
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (i + s) < numCities) {
            if (sdata[tid + s] < sdata[tid]) {
                sdata[tid] = sdata[tid + s];
                sIMin[tid] = sIMin[tid + s];
                sJMin[tid] = sJMin[tid + s];
            }
        }
        __syncthreads();
    }

    // The first thread in the block writes the block's result to global memory
    if (tid == 0) {
        atomicMin_double(globalMinChange, sdata[0], globalMinI, sIMin[0], globalMinJ, sJMin[0]);
    }
}


void twoOptCuda(const double* h_distanceMatrix, int* h_tour, int* h_candidateList, int numCities, int k) {
    // Device pointers
    double* d_distanceMatrix;
    int* d_tour;
    int* d_candidateList;
    int* d_positionMap;
    double* d_globalMinChanges;
    int* d_globalIMin;
    int* d_globalJMin;
    double* d_globalMinChange;
    int* d_globalMinI;
    int* d_globalMinJ;

    // Allocate memory on the GPU
    cudaMalloc(&d_distanceMatrix, numCities * numCities * sizeof(double));
    cudaMalloc(&d_tour, numCities * sizeof(int));
    cudaMalloc(&d_candidateList, numCities * k * sizeof(int));
    cudaMalloc(&d_positionMap, numCities * sizeof(int));
    cudaMalloc(&d_globalMinChanges, numCities * sizeof(double));
    cudaMalloc(&d_globalIMin, numCities * sizeof(int));
    cudaMalloc(&d_globalJMin, numCities * sizeof(int));
    cudaMalloc(&d_globalMinChange, sizeof(double));
    cudaMalloc(&d_globalMinI, sizeof(int));
    cudaMalloc(&d_globalMinJ, sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_distanceMatrix, h_distanceMatrix, numCities * numCities * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tour, h_tour, numCities * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_candidateList, h_candidateList, numCities * k * sizeof(int), cudaMemcpyHostToDevice);

    // Create and initialize the position map
    int* h_positionMap = new int[numCities];
    for (int i = 0; i < numCities; ++i) {
        h_positionMap[h_tour[i]] = i;
    }
    cudaMemcpy(d_positionMap, h_positionMap, numCities * sizeof(int), cudaMemcpyHostToDevice);
    delete[] h_positionMap;  // Clean up host memory

    // Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (numCities + threadsPerBlock - 1) / threadsPerBlock;

    bool improvement = true;
    while (improvement) {
        improvement = false;

        // Launch the 2-opt kernel to compute potential improvements
        twoOptKernel << <blocksPerGrid, threadsPerBlock >> > (d_distanceMatrix, d_tour, d_candidateList, d_positionMap, numCities, d_globalMinChanges, d_globalIMin, d_globalJMin, k);

        // Reset global minimum values before finding the minimum change
        cudaMemset(d_globalMinChange, DBL_MAX, sizeof(double));
        cudaMemset(d_globalMinI, -1, sizeof(int));
        cudaMemset(d_globalMinJ, -1, sizeof(int));

        // Launch the kernel to find the minimum change in the tour
        int sharedMemSize = threadsPerBlock * (sizeof(double) + 2 * sizeof(int));
        findMinChangeKernel << <blocksPerGrid, threadsPerBlock, sharedMemSize >> > (d_globalMinChanges, d_globalIMin, d_globalJMin, numCities, d_globalMinChange, d_globalMinI, d_globalMinJ);

        // Copy the results from device to host
        double h_globalMinChange;
        int h_globalMinI, h_globalMinJ;

        cudaMemcpy(&h_globalMinChange, d_globalMinChange, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_globalMinI, d_globalMinI, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_globalMinJ, d_globalMinJ, sizeof(int), cudaMemcpyDeviceToHost);

        // If there is an improvement, reverse the segment and continue
        if (h_globalMinI != -1 && h_globalMinJ != -1 && h_globalMinChange < 0) {
            int start = min(h_globalMinI, h_globalMinJ);
            int end = max(h_globalMinI, h_globalMinJ);

            // Launch the kernel to reverse the tour segment
            int threadsPerBlockReverse = 256;
            int blocksPerGridReverse = (end - start + threadsPerBlockReverse - 1) / threadsPerBlockReverse;
            reverseSegment << <blocksPerGridReverse, threadsPerBlockReverse >> > (d_tour, d_positionMap, start, end);

            improvement = true;
        }
    }

    // Copy the final tour from device to host
    cudaMemcpy(h_tour, d_tour, numCities * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_distanceMatrix);
    cudaFree(d_tour);
    cudaFree(d_candidateList);
    cudaFree(d_positionMap);
    cudaFree(d_globalMinChanges);
    cudaFree(d_globalIMin);
    cudaFree(d_globalJMin);
    cudaFree(d_globalMinChange);
    cudaFree(d_globalMinI);
    cudaFree(d_globalMinJ);
}



