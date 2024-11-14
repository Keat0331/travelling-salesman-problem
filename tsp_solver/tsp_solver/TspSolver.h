#pragma once
#ifndef TSPSOLVER_H
#define TSPSOLVER_H

#include "MurmurHash3.h"
#include <vector>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>

using namespace std;

struct City {
    double x, y;
};

struct VectorHash {
    size_t operator()(const vector<int>& vec) const {
        size_t hash = 0;
        MurmurHash3_x86_32(vec.data(), vec.size() * sizeof(int), 0, &hash);
        return hash;
    }
};

struct VectorEqual {
    bool operator()(const vector<int>& lhs, const vector<int>& rhs) const {
        return lhs == rhs;
    }
};

struct BestTour {
    vector<int> tour;
    double length = DBL_MAX;
    int generation = -1;
};

class TspSolver {
protected:
    vector<City> cities;              // List of cities
    vector<vector<int>> population;   // Population of tour solutions
    int populationSize;               // Size of the population
    double eliminationRate;           // Rate at which individuals are eliminated in each generation
    double mutationRate;              // Rate at which individuals are mutated
    mt19937 rng;                      // Random number generator
    unordered_set<vector<int>, VectorHash, VectorEqual> improvedTours;  // Improved tours
    vector<vector<double>> distanceMatrix; // Precomputed distance matrix
    vector<vector<int>> candidateList;     // Candidate list for 2-Opt++

    // Calculate Euclidean distance between two cities
    double distance(int i, int j);

    // Initialize the population with random tours
    void initializePopulation();

    // Perform natural selection on the population
    void naturalSelection();

    // Perform crossover between two parent tours to create a child tour
    vector<int> greedySubtourCrossover(const vector<int>& parent1, const vector<int>& parent2);

    // Create new individuals to replace the eliminated ones
    void multiply();

    // Apply 2-opt local search to improve a tour
    void twoOpt(vector<int>& tour);

    // Apply mutation (2-opt local search) to some individuals in the population
    void mutate();

    // Precompute distances between all pairs of cities
    void precomputeDistances();

    // Create candidate list for 2-Opt++
    void createCandidateList(int k);

    // Helper function to update position map after tour modification
    void updatePositionMap(vector<int>& tour, unordered_map<int, int>& positionMap, int start, int end);

public:
    // Constructor
    TspSolver(const vector<City>& cities, int populationSize, double eliminationRate, double mutationRate, unsigned int seed, int candidateSize);

    // Calculate the total length of a tour
    double tourLength(const vector<int>& tour);

    // Solve the TSP for a given number of generations
    BestTour solve(int generations, double diff);
};

#endif // TSPSOLVER_H


