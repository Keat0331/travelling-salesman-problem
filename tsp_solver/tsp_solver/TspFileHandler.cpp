#include "TspFileHandler.h"
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

// Function to read city coordinates from a TSP file
vector<City> readTspFile(const string& filename) {
    vector<City> cities;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return cities;
    }

    string line;
    // Skip lines until the NODE_COORD_SECTION header is found
    while (getline(file, line)) {
        if (line == "NODE_COORD_SECTION") break;
    }

    // Read city coordinates until the EOF header is found
    while (getline(file, line)) {
        if (line == "EOF") break;
        istringstream iss(line);
        int index;
        double x, y;
        if (iss >> index >> x >> y) {
            cities.push_back(City{ x, y });
        }
    }

    return cities;
}

// Function to write the best tour, tour length, and time taken to a file
void writeTourDataToFile(const string& filename, const vector<int>& bestTour, double length, double timeTaken) {
    ofstream file(filename);  // Use ofstream without ios::app to overwrite the file

    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << " for writing." << endl;
        return;
    }

    // Write the best tour
    file << "Best tour:\n";
    for (size_t i = 0; i < bestTour.size(); ++i) {
        file << bestTour[i];
        if (i != bestTour.size() - 1) {
            file << " ";
        }
    }
    file << "\n";

    // Write the tour length
    file << "Tour length:\n" << length << "\n";

    // Write the time taken
    file << "Time taken:\n" << timeTaken << "\n";

    file.close();
}
