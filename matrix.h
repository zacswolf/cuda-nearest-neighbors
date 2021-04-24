// matrix.h
#pragma once
#include <vector>
#include <iostream>

using namespace std;

class Matrix {
    private:
        vector<float> data;
        int numRows;
        int numCols;
    public:
        Matrix(vector<float> data, int numRows, int numCols);

        float get(int row, int col);

        // Removes and returns column from data
        Matrix popColumn(int columnIndex);

        void print();
};
