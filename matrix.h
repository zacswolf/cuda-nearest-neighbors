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

class Matrix {
	public:
		float *data;
		int numRows;
		int numCols;

		Matrix(float *data, int numRows, int numCols): data(data), numRows(numRows), numCols(numCols) { };
		Matrix(int numRows, int numCols): data(new float[numRows*numCols]), numRows(numRows), numCols(numCols) { };
		Matrix(vector<float> data, int numRows, int numCols): data(data.data()), numRows(numRows), numCols(numCols) { };


		int index(int row, int col);

		void fill(std::normal_distribution<float> distribution);

		// Removes and returns column from data
		pair<Matrix, Matrix> popColumn(int columnIndex);

		static Matrix matMulSeq(Matrix &left, Matrix &right);

		static float l2RowDistanceSeq(Matrix &left, int leftRow, Matrix &right, int rightRow);

		void print();

		~Matrix() {
			// delete [] data;
		}
};
