#include "matrix.h"

Matrix::Matrix(vector<float> data, int numRows, int numCols) {
	this->data = data;
	this->numRows = numRows;
	this->numCols = numCols;
}

float Matrix::get(int row, int col) {
	return this->data[row*(this->numCols) + col];
}

float Matrix::getNumRows() {
	return this->numRows;
}

float Matrix::getNumCols() {
	return this->numCols;
}

// Removes and returns column from data  
Matrix Matrix::popColumn(int columnIndex) {
	if (columnIndex < 0){
		columnIndex = this->numCols + columnIndex;
	}

	vector<float> data(this->numRows * (this->numCols - 1));
	vector<float> column(this->numRows);

	// Get specific elements from data and store in colummn
	for(int row = 0; row < this->numRows; row++) {
		column[row] = this->get(row, columnIndex);
	}

	// Copy this->data minus the popped column to a new data matrix

	// Copy first row up to columnIndex
	auto start = this->data.begin();
	auto end = this->data.begin() + columnIndex;
	auto destination = data.begin();
	copy(start, end, destination);

	for(int row = 1; row < this->numRows-1; row++) {
		// Adjust copy start and end as well as destination locations
		start = end+1;
		end += this->numCols;
		destination += this->numCols - 1;

		// Copy from [row-1, columnIndex+1] to (row, columnIndex)
		copy(start, end, destination);
	}

	// Adjust copy start and end as well as destination locations
	// Set end location to the end of the data matrix
	start = end+1;
	end = this->data.end();
	destination += this->numCols - 1;

	// Copy from [last row, columnIndex+1] to (last row, last column)
	copy(start, end, destination);

	this->data = data;
	this->numCols--;

	return Matrix(column, this->numRows, 1);
}

void Matrix::print() {
	printf("[\n");
	for (int row = 0; row < this->numRows; row++) {
		printf("[ ");
		for (int col = 0; col < this->numCols; col++) {
			printf("%f ", this->get(row, col));
		}
		printf("]\n");
	}
	printf("]\n");
}
