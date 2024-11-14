#include "TspSolverCuda.h"
#include "CudaKernel.cuh"
#include <numeric>
#include <cmath>
#include <algorithm>

using namespace std;

// Calculate Euclidean distance between two cities using the flattened distance matrix
double TspSolverCuda::distance(int i, int j) {
    int n = cities.size();
    return distanceMatrix[i * n + j];
}

// Initialize the population with random tours
void TspSolverCuda::initializePopulation() {
    vector<int> baseTour(cities.size());
    iota(baseTour.begin(), baseTour.end(), 0); // Initialize baseTour with 0, 1, 2, ..., cities.size()-1

    for (int i = 0; i < populationSize; i++) {
        shuffle(baseTour.begin(), baseTour.end(), rng);
        population.push_back(baseTour);
    }
}

// Perform natural selection on the population
void TspSolverCuda::naturalSelection() {
    int R = static_cast<int>(populationSize * eliminationRate);
    int64_t populationCount = population.size();
    vector<pair<double, int>> fitness(populationCount);

    // Calculate fitness values
    for (int i = 0; i < populationCount; ++i) {
        fitness[i] = { tourLength(population[i]), i };
    }

    // Sort the fitness values in ascending order
    sort(fitness.begin(), fitness.end());

    // Create a new population vector
    vector<vector<int>> newPopulation;
    newPopulation.reserve(populationCount - R);

    const double epsilon = 1e-6;  // Small positive real number
    int r = 0;

    // Use a vector<int> to keep track of elimination status
    vector<int> toEliminate(populationCount, 0);

    // Mark individuals to eliminate based on similarity in fitness
    for (int64_t i = 1; i < populationCount && r < R; ++i) {
        if (abs(fitness[i].first - fitness[i - 1].first) < epsilon) {
            toEliminate[i - 1] = 1;
            ++r;
        }
    }

    // If more individuals need to be eliminated, mark the ones with the worst fitness
    if (r < R) {
        int additionalToEliminate = R - r;
        for (int64_t i = populationCount - 1; i >= 0 && additionalToEliminate > 0; --i) {
            if (toEliminate[i] == 0) {
                toEliminate[i] = 1;
                --additionalToEliminate;
            }
        }
    }

    // Create the new population by excluding eliminated individuals
    for (int i = 0; i < populationCount; ++i) {
        if (toEliminate[i] == 0) {
            newPopulation.push_back(population[fitness[i].second]);
        }
    }

    // Move the new population back to the original population vector
    population = newPopulation;
}

// Perform crossover between two parent tours to create a child tour
vector<int> TspSolverCuda::greedySubtourCrossover(const vector<int>& parent1, const vector<int>& parent2) {
    vector<int> child;
    child.reserve(cities.size());
    int n = cities.size();
    vector<int> used(n, 0); // Vector to track used cities: 0 = not used, 1 = used
    bool fa = true, fb = true;

    // Choose town t randomly
    int64_t t = uniform_int_distribution<>(0, n - 1)(rng);
    child.push_back(t);
    used[t] = 1;

    // Find the position of t in both parents
    int64_t x = find(parent1.begin(), parent1.end(), t) - parent1.begin();
    int64_t y = find(parent2.begin(), parent2.end(), t) - parent2.begin();

    // Initialize part1 and part2 to hold segments before and after t
    vector<int> part1(n); // part1 will be filled in reverse order
    vector<int> part2(n); // part2 will be filled in forward order
    int i = 0, j = 0;

    while (fa || fb) {
        x = (x - 1 + parent1.size()) % parent1.size();  // Move x backwards in parent1
        y = (y + 1) % parent2.size();                   // Move y forwards in parent2

        if (fa && used[parent1[x]] == 0) {
            part1[i++] = parent1[x];
            used[parent1[x]] = 1;
        }
        else {
            fa = false;
        }

        if (fb && used[parent2[y]] == 0) {
            part2[j++] = parent2[y];
            used[parent2[y]] = 1;
        }
        else {
            fb = false;
        }
    }

    // Shrink part1 and part2 to fit their actual sizes
    part1.resize(i);
    part2.resize(j);

    // Reverse part1 because it was built in reverse order
    reverse(part1.begin(), part1.end());

    // Insert part1 at the beginning of child
    child.insert(child.begin(), part1.begin(), part1.end());

    // Insert part2 at the end of child
    child.insert(child.end(), part2.begin(), part2.end());

    // Add any remaining towns to child in random order
    if (child.size() < n) {
        vector<int> unused;
        for (int i = 0; i < n; i++) {
            if (used[i] == 0) {
                unused.push_back(i);
            }
        }
        shuffle(unused.begin(), unused.end(), rng);
        child.insert(child.end(), unused.begin(), unused.end());
    }

    return child;
}

// Create new individuals to replace the eliminated ones
void TspSolverCuda::multiply() {
    int64_t childrenCount = populationSize - population.size();

    // Resize population to the target size
    population.resize(populationSize);

    // Create new individuals and populate the vector
    int currentSize = population.size();
    for (int i = currentSize - childrenCount; i < currentSize; i++) {
        // Randomly select two parents
        int64_t parent1 = uniform_int_distribution<>(0, currentSize - childrenCount - 1)(rng);
        int64_t parent2 = uniform_int_distribution<>(0, currentSize - childrenCount - 1)(rng);

        // Add the new individual to the vector
        population[i] = greedySubtourCrossover(population[parent1], population[parent2]);
    }
}

// Apply mutation (2-opt local search) to some individuals in the population
void TspSolverCuda::mutate() {
    int mutationCount = static_cast<int>(population.size() * mutationRate);

    // Apply 2-opt to the best tour
    auto best = min_element(population.begin(), population.end(), [this](const vector<int>& a, const vector<int>& b) {
        return tourLength(a) < tourLength(b);
        });

    if (best != population.end() && improvedTours.find(*best) == improvedTours.end()) {
        twoOptCuda(distanceMatrix.data(), best->data(), flattenCandidateList.data(), cities.size(), candidateSize);
        improvedTours.insert(*best);
    }

    // Shuffle indices to randomly select individuals for mutation
    vector<int> indices(population.size());
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), rng);

    // Apply 2-opt mutation to selected individuals
    for (int i = 0; i < mutationCount; i++) {
        if (improvedTours.find(population[indices[i]]) == improvedTours.end()) {
            twoOptCuda(distanceMatrix.data(), population[indices[i]].data(), flattenCandidateList.data(), cities.size(), candidateSize);
            improvedTours.insert(population[indices[i]]);
        }
    }
}

// Precompute distances between all pairs of cities in the TSP solver
void TspSolverCuda::precomputeDistances() {
    int n = cities.size();  // Get the number of cities

    // Flatten the 2D distance matrix into a 1D vector for CUDA processing
    distanceMatrix.resize(n * n);

    // Call the CUDA function to compute distances and store them in distanceMatrix
    CudaComputeDistancesKernel(cities.data(), distanceMatrix.data(), n);
}

// Use a vector for the top-k closest cities
void TspSolverCuda::createCandidateList(int k) {
    flattenCandidateList.resize(cities.size() * k);
    int n = cities.size();  // Number of cities

    // Allocate host memory for closest cities
    vector<CityDistance> closestCities(n * k);

    // Call the CUDA function to find the top-k closest cities for all cities
    findTopKClosestCities(distanceMatrix.data(), closestCities.data(), n, k);

    // Populate the flattenCandidateList with the results
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            int cityIdx = closestCities[i * k + j].cityIdx;
            flattenCandidateList[i * k + j] = cityIdx;
        }
    }
}

// Constructor
TspSolverCuda::TspSolverCuda(const vector<City>& cities, int populationSize, double eliminationRate, double mutationRate, unsigned int seed, int candidateSize)
    : cities(cities), populationSize(populationSize), eliminationRate(eliminationRate), mutationRate(mutationRate), rng(seed), candidateSize(candidateSize) {
    precomputeDistances();
    createCandidateList(candidateSize);
}

// Calculate the total length of a tour
double TspSolverCuda::tourLength(const vector<int>& tour) {
    double length = 0.0;
    for (int i = 0; i < tour.size(); ++i) {
        length += distance(tour[i], tour[(i + 1) % tour.size()]);
    }
    return length;
}

// Solve the TSP for a given number of generations
BestTour TspSolverCuda::solve(int generations, double diff) {
    BestTour bestTour;
    initializePopulation();


    for (int i = 1; i <= generations; ++i) {
        naturalSelection();
        multiply();
        mutate();

        // find best tour
        auto localTour = min_element(population.begin(), population.end(), [this](const vector<int>& a, const vector<int>& b) {
            return tourLength(a) < tourLength(b);
            });

        double localLength = tourLength(*localTour);

        if (bestTour.length - localLength >= diff) {
            bestTour.tour = *localTour;
            bestTour.length = localLength;
            bestTour.generation = i;
        }
        else {
            break;
        }
    }

    // Return the best tour found
    return bestTour;
}



