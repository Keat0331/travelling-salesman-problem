#pragma once
#ifndef TSPFILEHANDLER_H
#define TSPFILEHANDLER_H

#include <vector>
#include <string>
#include "TspSolver.h" // Include TspSolver.h to access the City structure

std::vector<City> readTspFile(const std::string& filename);
void writeTourDataToFile(const std::string& filename, const std::vector<int>& bestTour, double length, double timeTaken);

#endif // TSPFILEHANDLER_H
