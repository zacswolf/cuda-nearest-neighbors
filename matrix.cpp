#include "matrix.h"

void Matrix::fill(std::normal_distribution<float> distribution) {
	std::default_random_engine generator;

	for (int i = 0; i < (this->numRows * this->numCols); i++) {
		this->data[i] = distribution(generator);
	}
}

void Matrix::fill(std::bernoulli_distribution distribution) {
	std::default_random_engine generator;

	for (int i = 0; i < (this->numRows * this->numCols); i++) {
		this->data[i] = distribution(generator);
	}
}

int Matrix::index(int row, int col) {
	return row*(this->numCols) + col;
}

// Removes and returns column from data  
pair<Matrix, Matrix> Matrix::popColumn(int columnIndex) {
	if (columnIndex < 0){
		columnIndex = this->numCols + columnIndex;
	}

	float *data = new float[this->numRows * (this->numCols - 1)];
	float *column = new float[this->numRows];

	// Get specific elements from data and store in colummn
	for(int row = 0; row < this->numRows; row++) {
		column[row] = this->data[this->index(row, columnIndex)];
	}

	// Copy this->data minus the popped column to a new data matrix

	// Copy first row up to columnIndex
	auto start = this->data;
	auto end = start + columnIndex;
	auto destination = data;
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
	end = this->data + (this->numRows * this->numCols);
	destination += this->numCols - 1;

	// Copy from [last row, columnIndex+1] to (last row, last column)
	copy(start, end, destination);

	// mat.numCols--;
	

	return make_pair(Matrix(column, this->numRows, 1), Matrix(data, this->numRows, this->numCols-1));
}

Matrix Matrix::matMulSeq(Matrix &left, Matrix &right) {
	int dim1 = left.numRows;
	int dim2 = left.numCols;
	int dim3 = right.numCols;
	assert(dim2 == right.numRows);

	Matrix result = Matrix(dim1, dim3);

	// Matrix Mult
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim3; j++) {
            for (int k = 0; k < dim2; k++) {
                result.data[result.index(i, j)] += left.data[left.index(i, k)] * right.data[right.index(k, j)];
			}
        }
    }
	
	return result;
}

float Matrix::l2RowDistanceSeq(Matrix &left, int leftRow, Matrix &right, int rightRow) {
	int dim = left.numCols;
	assert(dim == right.numCols);

	float currentDistance = 0.;
	for (int d = 0; d < dim; d++) {
		float term = left.data[left.index(leftRow, d)] - right.data[right.index(rightRow, d)];
		currentDistance += term*term;
	}

	return currentDistance;
}

void Matrix::print() {
	printf("[\n");
	for (int row = 0; row < this->numRows; row++) {
		printf("[ ");
		for (int col = 0; col < this->numCols; col++) {
			printf("%f ", this->data[this->index(row, col)]);
		}
		printf("]\n");
	}
	printf("]\n");
}
