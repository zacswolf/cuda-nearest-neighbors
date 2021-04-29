#include "matrix.h"

void fill(Matrix &mat, std::normal_distribution<float> distribution) {
	std::default_random_engine generator;

	for (int i = 0; i < (mat.numRows * mat.numCols); i++){
		mat.data[i] = distribution(generator);
	}
}

int index(Matrix &mat, int row, int col) {
	return row*(mat.numCols) + col;
}

// Removes and returns column from data  
pair<Matrix, Matrix> popColumn(Matrix &mat, int columnIndex) {
	if (columnIndex < 0){
		columnIndex = mat.numCols + columnIndex;
	}


	vector<float> data(mat.numRows * (mat.numCols - 1));
	vector<float> column(mat.numRows);

	// Get specific elements from data and store in colummn
	for(int row = 0; row < mat.numRows; row++) {
		column[row] = mat.data[index(mat, row, columnIndex)];
	}

	// Copy this.data minus the popped column to a new data matrix

	// Copy first row up to columnIndex
	auto start = mat.data.begin();
	auto end = mat.data.begin() + columnIndex;
	auto destination = data.begin();
	copy(start, end, destination);

	for(int row = 1; row < mat.numRows-1; row++) {
		// Adjust copy start and end as well as destination locations
		start = end+1;
		end += mat.numCols;
		destination += mat.numCols - 1;

		// Copy from [row-1, columnIndex+1] to (row, columnIndex)
		copy(start, end, destination);
	}

	// Adjust copy start and end as well as destination locations
	// Set end location to the end of the data matrix
	start = end+1;
	end = mat.data.end();
	destination += mat.numCols - 1;

	// Copy from [last row, columnIndex+1] to (last row, last column)
	copy(start, end, destination);

	// mat.numCols--;
	

	return make_pair(Matrix({column, mat.numRows, 1}), Matrix({data, mat.numRows, mat.numCols-1}));
}
void print(Matrix &mat) {
	printf("[\n");
	for (int row = 0; row < mat.numRows; row++) {
		printf("[ ");
		for (int col = 0; col < mat.numCols; col++) {
			printf("%f ", mat.data[index(mat,row, col)]);
		}
		printf("]\n");
	}
	printf("]\n");
}
