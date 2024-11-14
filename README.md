# Traveling Salesman Problem (TSP) Solver

This project addresses the Traveling Salesman Problem (TSP) using Genetic Algorithms implemented in C++. It also includes parallelized versions using CUDA and OpenMP.

## Project Structure
- **`Graph_Verification/`**: Contains the optimal results for the TSP problem instances used.
- **`tsp_solver/`**: Includes the main genetic algorithm implementations:
  - **`TspSolver.cpp`**: Single-threaded version.
  - **`TspSolverOmp.cpp`**: OpenMP-parallelized version for multi-core CPUs.
  - **`TspSolverCuda.cu`**: CUDA-parallelized version for GPUs.

## Implementation Details
- **Genetic Algorithm**: The core algorithm used to solve the TSP. This implementation is not fully optimized but serves as a solid reference.
- **Parallelization**:
  - **CUDA**: Uses GPU acceleration for faster computation.
  - **OpenMP**: Multithreaded CPU version for improved processing time on multi-core systems.

## How to Run
1. Use Visual Studio to run `main.cpp` via the command line interface (CLI).
2. For CUDA and OpenMP versions, ensure your system has the necessary support for these parallelization libraries.

## Note
This algorithm is intended as a reference and is not fully optimized.
