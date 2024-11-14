#include "TspSolverOmp.h"
#include <numeric>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <omp.h>

using namespace std;

// Calculate Euclidean distance between two cities using the precomputed distance matrix
double TspSolverOmp::distance(int i, int j) {
    return distanceMatrix[i][j];
}

// Initialize the population with random tours
void TspSolverOmp::initializePopulation() {
    population.clear();
    vector<int> baseTour(cities.size());
    iota(baseTour.begin(), baseTour.end(), 0); // Initialize baseTour with 0, 1, 2, ..., cities.size()-1

    // Create a temporary population to avoid race conditions
    vector<vector<int>> tempPopulation(populationSize);

    // let each thread create a base tour for each individual in the population
    #pragma omp parallel
    {
        // Create a thread-local RNG
        random_device rd;
        mt19937 localRng(rd());

        #pragma omp for
        for (int i = 0; i < populationSize; i++) {
            vector<int> localTour = baseTour;
            shuffle(localTour.begin(), localTour.end(), localRng); // Use the thread-local RNG
            tempPopulation[i] = localTour;
        }
    }

    // Copy the temporary population to the actual population
    population = tempPopulation;
}

// Perform natural selection on the population
void TspSolverOmp::naturalSelection() {
    int R = static_cast<int>(populationSize * eliminationRate);
    int64_t populationCount = population.size();
    vector<pair<double, int>> fitness(populationCount);

    for (int i = 0; i < populationCount; ++i) {
        fitness[i] = { tourLength(population[i]), i };
    }

    // Sort the fitness values in ascending order
    sort(fitness.begin(), fitness.end());

    // Create a new population vector
    vector<vector<int>> newPopulation;
    newPopulation.reserve(populationCount - R);

    const double epsilon = 1e-6;  // Small positive real number
    vector<int> toEliminate(populationCount, 0); // 0 means not eliminated, 1 means eliminated

    atomic<int> r(0);

    // Step 2: Parallel marking for elimination based on fitness similarity
    #pragma omp parallel for
    for (int64_t i = 1; i < populationCount; ++i) {
        if (abs(fitness[i].first - fitness[i - 1].first) < epsilon) {
            if (r.fetch_add(1) < R) { // Atomically increment r and check its value
                toEliminate[i - 1] = 1; // Mark as eliminated
            }
            else {
                r--; // If r exceeded R, decrement it back
            }
        }
    }

    // If more individuals need to be eliminated, mark the ones with the worst fitness
    if (r < R) {
        int additionalToEliminate = R - r;
        for (int64_t i = populationCount - 1; i >= 0 && additionalToEliminate > 0; --i) {
            if (toEliminate[i] == 0) { // Check if not already eliminated
                toEliminate[i] = 1; // Mark as eliminated
                --additionalToEliminate;
            }
        }
    }

    // Step 3: Parallel construction of the new population
    #pragma omp parallel 
    {
        vector<vector<int>> localNewPopulation;
        localNewPopulation.reserve(populationCount - R);

        #pragma omp for nowait
        for (int i = 0; i < populationCount; ++i) {
            if (toEliminate[i] == 0) { // Include only non-eliminated individuals
                localNewPopulation.push_back(population[fitness[i].second]);
            }
        }

        // Merge the local results into the main newPopulation
        #pragma omp critical
        {
            newPopulation.insert(newPopulation.end(), localNewPopulation.begin(), localNewPopulation.end());
        }
    }

    // Move the sorted new population back to the original population vector
    population = newPopulation;
}

// Perform crossover between two parent tours to create a child tour
vector<int> TspSolverOmp::greedySubtourCrossover(const vector<int>& parent1, const vector<int>& parent2, mt19937 localRng) {
    vector<int> child;
    child.reserve(cities.size());
    int n = cities.size();
    vector<int> used(n, 0); // Vector to track used cities: 0 = not used, 1 = used
    bool fa = true, fb = true;

    // Choose town t randomly
    int64_t t = uniform_int_distribution<>(0, n - 1)(localRng);
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
        shuffle(unused.begin(), unused.end(), localRng);
        child.insert(child.end(), unused.begin(), unused.end());
    }

    return child;
}

// Create new individuals to replace the eliminated ones
void TspSolverOmp::multiply() {
    int64_t childrenCount = populationSize - population.size();

    // Resize population to the target size
    population.resize(populationSize);

    int currentSize = population.size();

    #pragma omp parallel
    {
        // Each thread needs its own random number generator to avoid contention
        random_device rd;
        mt19937 localRng(rd()); // Initialize local RNG

        #pragma omp for
        for (int64_t i = currentSize - childrenCount; i < currentSize; i++) {
            // Randomly select two parents
            uniform_int_distribution<> dist(0, currentSize - childrenCount - 1);
            int64_t parent1 = dist(localRng);
            int64_t parent2 = dist(localRng);

            // Add the new individual to the vector
            population[i] = greedySubtourCrossover(population[parent1], population[parent2], localRng);
        }
    }
}

// Apply 2-opt local search to improve a tour
void TspSolverOmp::twoOpt(vector<int>& tour) {
    unordered_map<int, int> positionMap;
    for (int i = 0; i < tour.size(); ++i) {
        positionMap[tour[i]] = i;
    }

    bool improved = true;
    int n = tour.size();

    while (improved) {
        improved = false;

        // Iterate over each city
        for (int i = 0; i < n; ++i) {
            int a1 = tour[i]; // Current city  

            // Get the candidate list for the current city
            const vector<int>& candidates = candidateList[a1];

            // Check each candidate
            for (int j = 0; j < candidates.size(); j++) {
                int a2 = tour[(positionMap[a1] + 1) % n]; // Next city in the tour
                int b1 = candidates[j];  // city b1

                if (a2 == candidates[0]) break; // Early exit if a2 == nearest city

                int b2 = tour[(positionMap[b1] + 1) % n];  // next of city b1

                if (a2 == b1 || b2 == a1) continue; // Skip if cities are adjacent

                // Calculate distances
                double ab = distance(a1, a2);
                double cd = distance(b1, b2);
                double ac = distance(a1, b1);
                double bd = distance(a2, b2);

                if (ab + cd > ac + bd) {
                    if (positionMap[b1] > positionMap[a1]) {
                        reverse(tour.begin() + positionMap[a1] + 1, tour.begin() + positionMap[b1] + 1);
                        updatePositionMap(tour, positionMap, positionMap[a1], positionMap[b1]);
                    }
                    else {
                        reverse(tour.begin() + positionMap[b1] + 1, tour.begin() + positionMap[a1] + 1);
                        updatePositionMap(tour, positionMap, positionMap[b1], positionMap[a1]);
                    }


                    improved = true;
                }
            }
        }
    }
}

// Optimized helper function to update position map after tour modification
void TspSolverOmp::updatePositionMap(vector<int>& tour, unordered_map<int, int>& positionMap, int start, int end) {
    for (int i = start + 1; i <= end; ++i) {
        positionMap[tour[i]] = i;
    }
}

// Apply mutation (2-opt local search) to some individuals in the population
void TspSolverOmp::mutate() {
    int mutationCount = static_cast<int>(population.size() * mutationRate);

    // Apply 2-opt to the best tour
    auto best = min_element(population.begin(), population.end(), [this](const vector<int>& a, const vector<int>& b) {
        return tourLength(a) < tourLength(b);
        });

    if (best != population.end() && improvedTours.find(*best) == improvedTours.end()) {
        twoOpt(*best);
        improvedTours.insert(*best);
    }

    // Shuffle indices to randomly select individuals for mutation
    vector<int> indices(population.size());
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), rng);

    // each thread run one two opt for one tour in parallel
    #pragma omp parallel for 
    for (int i = 0; i < mutationCount; i++) {
        vector<int> localTour = population[indices[i]]; // Create a thread-local copy
        if (improvedTours.find(localTour) == improvedTours.end()) {
            twoOpt(localTour);

            #pragma omp critical
            {
                improvedTours.insert(localTour);
            }

            population[indices[i]] = localTour; // Update the original population with improved tour
        }
    }

}

// Precompute distances between all pairs of cities
void TspSolverOmp::precomputeDistances() {
    int64_t  n = cities.size();
    distanceMatrix.resize(n, vector<double>(n));

    // just treat the matrix as 1d vector by joining the row together, each thread calculating the distance for each column
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            distanceMatrix[i][j] = sqrt(pow(cities[i].x - cities[j].x, 2) + pow(cities[i].y - cities[j].y, 2));
        }
    }
}

// Use a vector for the top-k closest cities
void TspSolverOmp::createCandidateList(int k) {
    candidateList.resize(cities.size());

    // let each thread handle each city in parallel
    #pragma omp parallel for
    for (int i = 0; i < cities.size(); ++i) {
        vector<pair<double, int>> closestCities(k, { std::numeric_limits<double>::max(), -1 });

        for (int j = 0; j < cities.size(); ++j) {
            if (i != j) {
                double dist = distance(i, j);

                // Insert the city into the sorted array of closest cities
                for (int l = 0; l < k; ++l) {
                    if (dist < closestCities[l].first) {
                        // Shift the elements to the right to make space
                        for (int m = k - 1; m > l; --m) {
                            closestCities[m] = closestCities[m - 1];
                        }
                        closestCities[l] = { dist, j };
                        break;
                    }
                }
            }
        }

        for (int l = 0; l < k; ++l) {
            if (closestCities[l].second != -1) {
                candidateList[i].push_back(closestCities[l].second);
            }
        }
    }
}

// Constructor
TspSolverOmp::TspSolverOmp(const vector<City>& cities, int populationSize, double eliminationRate, double mutationRate, unsigned int seed, int candidateSize)
    : cities(cities), populationSize(populationSize), eliminationRate(eliminationRate), mutationRate(mutationRate), rng(seed) {
    precomputeDistances();
    createCandidateList(candidateSize);
}

// Calculate the total length of a tour (combination of cities)
double TspSolverOmp::tourLength(const vector<int>& tour) {
    double length = 0.0;

    // each thread calculate the respective partial length and then add up to be the actual length
    #pragma omp parallel for reduction(+:length)
    for (int i = 0; i < tour.size(); i++) {
        length += distance(tour[i], tour[(i + 1) % tour.size()]);
    }
    return length;
}

// Solve the TSP for a given number of generations
BestTour TspSolverOmp::solve(int generations, int numberOfThreads, double diff) {
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


