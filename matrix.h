// matrix.h
#pragma once
#include <vector>
#include <iostream>
#include "random"
#include "vector"
#include "utility"

#include "matrix_interface.h"
#include "exceptions.h"

using namespace std;

struct Matrix {
	vector<float> data;

	int numRows;
	int numCols;
};

void fill(Matrix &mat, std::normal_distribution<float> distribution);

int index(Matrix &mat, int row, int col);

// Removes and returns column from data
pair<Matrix, Matrix> popColumn(Matrix &mat, int columnIndex);

void print(Matrix &mat);
