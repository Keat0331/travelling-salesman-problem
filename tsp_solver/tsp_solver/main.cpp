#include "TspSolver.h"
#include "TspSolverOmp.h"
#include "TspSolverCuda.h"
#include "TspFileHandler.h"
#include <iostream>
#include <chrono>
#include <numeric>
#include <cstdlib> 
#include <omp.h>

using namespace std;

double st_tsp(string outFilename, vector<City> cities, unsigned int seed,
    int populationSize, double eliminationRate, double mutationRate, int generations, int candidateSize, double diff) {

    // Create a TspSolver instance
    TspSolver solver(cities, populationSize, eliminationRate, mutationRate, seed, candidateSize);

    // Measure time taken
    auto start = chrono::high_resolution_clock::now();
    BestTour bestTour = solver.solve(generations, diff);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    // Calculate tour length
    double bestLength = bestTour.length;

    // Write the best tour, tour length, and time taken to a file
    writeTourDataToFile(outFilename, bestTour.tour, bestLength, duration.count());

    cout << "\nOriginal best tour found on generation: " << bestTour.generation << endl;
    cout << "\nOriginal tour length: " << bestLength << endl;
    cout << "\nOriginal time taken: " << duration.count() << " seconds" << endl << endl;

    return duration.count();
}

double omp_tsp(string outFilename, vector<City> cities, unsigned int seed,
    int populationSize, double eliminationRate, double mutationRate, int generations, int candidateSize, int numberOfThreads, double diff) {

    // Create a TspSolverOmp instance
    TspSolverOmp solver(cities, populationSize, eliminationRate, mutationRate, seed, candidateSize);

    // Measure time taken
    auto start = chrono::high_resolution_clock::now();
    BestTour bestTour = solver.solve(generations, numberOfThreads, diff);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    // Calculate tour length
    double bestLength = bestTour.length;

    // Write the best tour, tour length, and time taken to a file
    writeTourDataToFile(outFilename, bestTour.tour, bestLength, duration.count());

    cout << "\nOMP best tour found on generation: " << bestTour.generation << endl;
    cout << "\nOMP tour length: " << bestLength << endl;
    cout << "\nOMP time taken: " << duration.count() << " seconds" << endl << endl;

    return duration.count();
}

double cuda_tsp(string outFilename, vector<City> cities, unsigned int seed,
    int populationSize, double eliminationRate, double mutationRate, int generations, int candidateSize, double diff) {

    // Create a TspSolver instance
    TspSolverCuda solver(cities, populationSize, eliminationRate, mutationRate, seed, candidateSize);

    // Measure time taken
    auto start = chrono::high_resolution_clock::now();
    BestTour bestTour = solver.solve(generations, diff);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    // Calculate tour length
    double bestLength = bestTour.length;

    // Write the best tour, tour length, and time taken to a file
    writeTourDataToFile(outFilename, bestTour.tour, bestLength, duration.count());

    cout << "\nCUDA best tour found on generation: " << bestTour.generation << endl;
    cout << "\nCUDA tour length: " << bestLength << endl;
    cout << "\nCUDA time taken: " << duration.count() << " seconds" << endl << endl;

    return duration.count();
}


int main(int argc, char* argv[]) {
    // Check if the correct number of arguments is provided
    if (argc != 10) {
        cerr << "Usage: " << argv[0] << " <TSP filename> <seed> <populationSize> <eliminationRate> <mutationRate> <generations> <candidateSize> <numberOfThreads> <difference>" << endl;
        return 1;
    }

    // Get the input file name from command line arguments
    string filename = "tsp_solver/data/" + string(argv[1]) + ".tsp";

    // Set output filenames based on the first argument
    string outFilename_st = "tsp_solver/st_result/" + string(argv[1]) + ".txt";
    string outFilename_omp = "tsp_solver/omp_result/" + string(argv[1]) + ".txt";
    string outFilename_cuda = "tsp_solver/cuda_result/" + string(argv[1]) + ".txt";

    // Read the cities from the TSP file
    vector<City> cities = readTspFile(filename);

    // Check if cities were successfully read
    if (cities.empty()) {
        cerr << "Error: No cities read from the file or file is empty." << endl;
        return 1;
    }

    // Convert command line arguments to the required types
    unsigned int seed = static_cast<unsigned int>(atoi(argv[2]));
    int populationSize = atoi(argv[3]);
    double eliminationRate = atof(argv[4]);
    double mutationRate = atof(argv[5]);
    int generations = atoi(argv[6]);
    int candidateSize = atoi(argv[7]);
    int numberOfThreads = atoi(argv[8]);
    double diff = atof(argv[9]);

    // Call the different TSP solvers with the provided arguments
    double time_st = st_tsp(outFilename_st, cities, seed, populationSize, eliminationRate, mutationRate, generations, candidateSize, diff);
    double time_omp = omp_tsp(outFilename_omp, cities, seed, populationSize, eliminationRate, mutationRate, generations, candidateSize, numberOfThreads, diff);
    double time_cuda = cuda_tsp(outFilename_cuda, cities, seed, populationSize, eliminationRate, mutationRate, generations, candidateSize, diff);

    cout << "\nperformance gain of omp = " << time_st / time_omp  << endl;
    cout << "performance gain of cuda = " << time_st / time_cuda << endl << endl;

    return 0;
}


// dataset
// -----------------------------------------------------------
// qa194, pbm436, lu980, tz6117, fi10639

// use the powershell in the visual studio to run the program
// -----------------------------------------------------------
// parameter [1] = TSP filename      (tsp file to be tested)
// parameter [2] = seed              (fix the random numbers, only applicable to serial code)
// parameter [3] = population Size   (size of the population)
// parameter [4] = elimination Rate  (specify how many percent of the tours to be removed from the population and generated during multiplication)
// parameter [5] = mutation Rate     (specify how many percent of the tours undergo two opt)
// parameter [6] = generations       (maximum number of generation)
// parameter [7] = candidate Size    (number of closest cities for each city)
// parameter [8] = number Of Threads (number of threads used by omp)
// parameter [9] = difference        (minimum difference of the best tours as an indicator on whether to continue the generations)
// 
// below are examples of command for different data
// -----------------------------------------------------------
// x64/Debug/tsp_solver.exe qa194 42 200 0.3 0.2 500 20 6 1
// x64/Debug/tsp_solver.exe pbm436 42 200 0.3 0.2 500 20 6 1
// x64/Debug/tsp_solver.exe lu980 42 200 0.3 0.2 500 20 6 1
// x64/Debug/tsp_solver.exe tz6117 42 200 0.3 0.2 500 20 6 1
// x64/Debug/tsp_solver.exe fi10639 42 200 0.3 0.2 500 20 6 1
