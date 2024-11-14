#include "TspSolver.h"
#include <numeric>
#include <cmath>
#include <algorithm>

using namespace std;

// Calculate Euclidean distance between two cities using the precomputed distance matrix
double TspSolver::distance(int i, int j) {
    return distanceMatrix[i][j];
}

// Initialize the population with random tours
void TspSolver::initializePopulation() {
    vector<int> baseTour(cities.size());
    iota(baseTour.begin(), baseTour.end(), 0); // Initialize baseTour with 0, 1, 2, ..., cities.size()-1

    for (int i = 0; i < populationSize; i++) {
        shuffle(baseTour.begin(), baseTour.end(), rng);
        population.push_back(baseTour);
    }
}

// Perform natural selection on the population
void TspSolver::naturalSelection() {
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
vector<int> TspSolver::greedySubtourCrossover(const vector<int>& parent1, const vector<int>& parent2) {
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
void TspSolver::multiply() {
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

// Apply 2-opt local search to improve a tour
void TspSolver::twoOpt(vector<int>& tour) {
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

    improvedTours.insert(tour);
}

// Optimized helper function to update position map after tour modification
void TspSolver::updatePositionMap(vector<int>& tour, unordered_map<int, int>& positionMap, int start, int end) {
    for (int i = start + 1; i <= end; ++i) {
        positionMap[tour[i]] = i;
    }
}

// Apply mutation (2-opt local search) to some individuals in the population
void TspSolver::mutate() {
    int mutationCount = static_cast<int>(population.size() * mutationRate);

    // Apply 2-opt to the best tour
    auto best = min_element(population.begin(), population.end(), [this](const vector<int>& a, const vector<int>& b) {
        return tourLength(a) < tourLength(b);
        });

    if (best != population.end() && improvedTours.find(*best) == improvedTours.end()) {
        twoOpt(*best);
    }

    // Shuffle indices to randomly select individuals for mutation
    vector<int> indices(population.size());
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), rng);

    // can be parallelized
    for (int i = 0; i < mutationCount; i++) {
        if (improvedTours.find(population[indices[i]]) == improvedTours.end()) {
            twoOpt(population[indices[i]]);
        }
    }
}

// Precompute distances between all pairs of cities
void TspSolver::precomputeDistances() {
    int n = cities.size();
    distanceMatrix.resize(n, vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            distanceMatrix[i][j] = sqrt(pow(cities[i].x - cities[j].x, 2) + pow(cities[i].y - cities[j].y, 2));
        }
    }
}

// Use a vector for the top-k closest cities
void TspSolver::createCandidateList(int k) {
    candidateList.resize(cities.size());

    // can be parallelized
    for (int i = 0; i < cities.size(); ++i) {
        vector<pair<double, int>> closestCities(k, { numeric_limits<double>::max(), -1 });

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
TspSolver::TspSolver(const vector<City>& cities, int populationSize, double eliminationRate, double mutationRate, unsigned int seed, int candidateSize)
    : cities(cities), populationSize(populationSize), eliminationRate(eliminationRate), mutationRate(mutationRate), rng(seed) {
    precomputeDistances();
    createCandidateList(candidateSize);
}

// Calculate the total length of a tour
double TspSolver::tourLength(const vector<int>& tour) {
    double length = 0.0;
    for (int i = 0; i < tour.size(); ++i) {
        length += distance(tour[i], tour[(i + 1) % tour.size()]);
    }
    return length;
}

// Solve the TSP for a given number of generations
BestTour TspSolver::solve(int generations, double diff) {
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



