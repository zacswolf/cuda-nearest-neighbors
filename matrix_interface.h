// matrix.h
#pragma once

using namespace std;

class MatrixInterface {
	public:
		// virtual MatrixInterface(vector<float> data, int numRows, int numCols) {};

		const virtual float get(int row, int col) = 0;

		virtual float getNumRows() = 0;

		virtual float getNumCols() = 0;

		// Removes and returns column from data
		virtual MatrixInterface* popColumn(int columnIndex) = 0;

		virtual void print() = 0;

        virtual MatrixInterface* to(int device) = 0;

        virtual ~MatrixInterface() {};
};
