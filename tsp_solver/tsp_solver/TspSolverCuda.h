#pragma once
#ifndef TSPSOLVERCUDA_H
#define TSPSOLVERCUDA_H

#include "TspSolver.h"
#include <vector>
#include <random>
#include <unordered_set>
#include <functional>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>

using namespace std;

class TspSolverCuda {
protected:
    vector<City> cities;              // List of cities
    vector<vector<int>> population;   // Population of tour solutions
    int populationSize;               // Size of the population
    double eliminationRate;           // Rate at which individuals are eliminated in each generation
    double mutationRate;              // Rate at which individuals are mutated
    mt19937 rng;                      // Random number generator
    unordered_set<vector<int>, VectorHash, VectorEqual> improvedTours;  // Improved tours
    vector<double> distanceMatrix; // Precomputed distance matrix
    vector<int> flattenCandidateList;
    int candidateSize;

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

    // Apply mutation (2-opt local search) to some individuals in the population
    void mutate();

    // Precompute distances between all pairs of cities
    void precomputeDistances();

    // Create candidate list for 2-Opt++
    void createCandidateList(int k);

public:
    // Constructor
    TspSolverCuda(const vector<City>& cities, int populationSize, double eliminationRate, double mutationRate, unsigned int seed, int candidateSize);

    // Calculate the total length of a tour
    double tourLength(const vector<int>& tour);

    // Solve the TSP for a given number of generations
    BestTour solve(int generations, double diff);
};

#endif // TSPSOLVERCUDA_H


