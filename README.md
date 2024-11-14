# Traveling Salesman Problem (TSP) Solver

This project addresses the Traveling Salesman Problem (TSP) using Genetic Algorithms implemented in C++. It also includes parallelized versions using CUDA and OpenMP.

## Project Structure
- **`graph_verification/`**: Contains the optimal results for the TSP problem instances used.
- **'tsp_solver/'**: The main genetic algorithm implementation and parallelized versions are located in the source folder.

## Implementation Details
- **Genetic Algorithm**: The primary solution for solving TSP. This implementation is not fully optimized, but it serves as a useful reference.
- **Parallelization**:
  - **CUDA**: GPU-accelerated version for faster computation.
  - **OpenMP**: Multithreaded CPU version to improve processing time on multi-core systems.

## How to Run
1. use Visual Studio to run it.
2. For CUDA and OpenMP, ensure your system supports these parallelization libraries.

## Note
This algorithm is still in a reference state and is not fully optimized.


