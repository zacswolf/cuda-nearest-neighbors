#include "matrix.h"

template <typename T>
__host__ void Matrix<T>::fill(std::normal_distribution<float> distribution) {
	std::default_random_engine generator;

	for (int i = 0; i < (this->numRows * this->numCols); i++) {
		this->data[i] = distribution(generator);
	}
}

template <typename T>
__host__ void Matrix<T>::fill(std::bernoulli_distribution distribution) {
	std::default_random_engine generator;

	for (int i = 0; i < (this->numRows * this->numCols); i++) {
		this->data[i] = distribution(generator);
	}
}

template <typename T>
__host__ __device__ int Matrix<T>::index(int row, int col) {
	return row*(this->numCols) + col;
}

// Removes and returns column from data  
template <typename T>
__host__ pair<Matrix<T>, Matrix<T>> Matrix<T>::popColumn(int columnIndex) {
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

template <typename T>
template <typename G>
__host__ Matrix<decltype(std::declval<T&>() * std::declval<G&>())> Matrix<T>::matMulSeq(Matrix<T> &left, Matrix<G> &right) {
	int dim1 = left.numRows;
	int dim2 = left.numCols;
	int dim3 = right.numCols;
	assert(dim2 == right.numRows);

	Matrix result = Matrix<decltype(std::declval<T&>() * std::declval<G&>())>(dim1, dim3);

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

template <typename T>
__host__ __device__ float Matrix<T>::l2RowDistanceSeq(Matrix &left, int leftRow, Matrix &right, int rightRow) {
	int dim = left.numCols;
	assert(dim == right.numCols);

	float currentDistance = 0.;
	for (int d = 0; d < dim; d++) {
		float term = left.data[left.index(leftRow, d)] - right.data[right.index(rightRow, d)];
		currentDistance += term*term;
	}

	return currentDistance;
}

template <typename T>
__host__ void Matrix<T>::print() {
	if (this->numCols != 1) {
		printf("[\n");
		for (int row = 0; row < this->numRows; row++) {
			printf("[ ");
			for (int col = 0; col < this->numCols; col++) {
				cout << this->data[this->index(row, col)] << " ";
			}
			printf("]\n");
		}
		printf("]\n");
	} else {
		printf("[");
		for (int row = 0; row < this->numRows; row++) {
			cout << this->data[this->index(row, 0)] << " ";
		}
		printf("]\n");
	}
}

// template class Matrix<float>;
// template class Matrix<bool>;
// template class Matrix<int>;
