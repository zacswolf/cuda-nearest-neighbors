// matrix.h
#pragma once
#include <vector>
#include <iostream>
#include "random"

#include "matrix_interface.h"
#include "exceptions.h"


using namespace std;

// class MatrixSlice {
// 	private:
// 		float *row;
// 	public:
// 		MatrixSlice(float * row) : row(row) {}
// 		float operator [](int i) const {return row[i];}
// }

class Matrix : public MatrixInterface {
	private:
		vector<float> data;
		int numRows;
		int numCols;
	public:
		Matrix(int numRows, int numCols);

		Matrix(vector<float> data, int numRows, int numCols);

		void fill(std::normal_distribution<float> distribution);

		const float get(int row, int col);
		void set(int row, int col, float val);
		float* operator [](int i) {return &(this->get(i,0));}

		float getNumRows();
		float getNumCols();

		// Removes and returns column from data
		Matrix* popColumn(int columnIndex);

		void print();

		MatrixInterface* to(int device);

		~Matrix() {};
};


