// matrix.h
#pragma once
#include <vector>
#include <iostream>
#include "matrix_interface.h"
#include "exceptions.h"

using namespace std;

class Matrix : public MatrixInterface {
	private:
		vector<float> data;
		int numRows;
		int numCols;
	public:
		Matrix(vector<float> data, int numRows, int numCols);

		float get(int row, int col);

		float getNumRows();

		float getNumCols();

		// Removes and returns column from data
		Matrix* popColumn(int columnIndex);

		void print();

		MatrixInterface* to(int device);

		~Matrix();
};
