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

// Move matrix between CPU and device
template <typename T>
__host__ Matrix<T> Matrix<T>::toDevice(int device) {
	if (this->device == 0) {
		assert(device != this->device);

		int dataBytes = (this->numRows * this->numCols) * sizeof(T);
		//cudaMalloc(&d_matrix, sizeof(Matrix<T>));
		//cudaMemcpy(&d_matrix, this, sizeof(Matrix<T>), cudaMemcpyHostToDevice);
		// Copy over data as well
		T *dataRaw;
		cudaMalloc(&dataRaw, dataBytes);
		cudaMemcpy(dataRaw, this->data, dataBytes, cudaMemcpyHostToDevice);
		// Set device data pointers
		//cudaMemcpy((void *)&(this->data), &dataRaw, sizeof(T *), cudaMemcpyHostToDevice);

		return Matrix<T>(dataRaw, this->numRows, this->numCols, device);
	} else if (this->device != 0 && device == 0) {
		// Move back to CPU
		assert(device != this->device);

		int dataBytes = (this->numRows * this->numCols) * sizeof(T);
		T *dataRaw = new T[this->numRows * this->numCols];
		cudaMemcpy(this->data, dataRaw, dataBytes, cudaMemcpyDeviceToHost);

		return Matrix<T>(dataRaw, this->numRows, this->numCols, device);
	} else {
		throw NotImplementedException("Matrix<T>::toDevice() for non-zero device");
	}
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

template <typename T, typename G>
__global__ void matMulGPUKernel(Matrix<T> left, Matrix<G> right, Matrix<decltype(std::declval<T&>() * std::declval<G&>())> result, int dimLeft, int dimRight, int dimCenter) {
	int a = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y * blockDim.y + threadIdx.y;

	//printf("i: %d, j%d\n", i, j);

	/*
	for (int k = 0; k < dimCenter; k++) {
		result.data[result.index(i, j)] += left.data[left.index(i, k)] * right.data[right.index(k, j)];
	}
	*/
	if (a == 0 && b == 0) {
		for (int i = 0; i < dimLeft; i++) {
			for (int j = 0; j < dimRight; j++) {
				for (int k = 0; k < dimCenter; k++) {
					result.data[result.index(i, j)] += left.data[left.index(i, k)] * right.data[right.index(k, j)];
				}
			}
		}
	}
}

template <typename T>
template <typename G>
__host__ Matrix<decltype(std::declval<T&>() * std::declval<G&>())> Matrix<T>::matMulGPU(Matrix<T> &left, Matrix<G> &right) {
	int dimLeft = left.numRows;
	int dimCenter = left.numCols;
	int dimRight = right.numCols;
	assert(dimCenter == right.numRows);

	assert(left.device == right.device);
	assert(left.device != 0);

	//Matrix result = Matrix<decltype(std::declval<T&>() * std::declval<G&>())>(dimLeft, dimRight, left.device);

	// Launching a 2D kernel
	int xBlock = (int)ceil(((float)dimLeft/512.0f));
	int yBlock = (int)ceil(((float)dimRight/512.0f));
	printf("block size should be: %d %d, dimLeft: %d, dimRight: %d\n", xBlock, yBlock, dimLeft, dimRight);
	dim3 blockSize(xBlock, yBlock);
	int bx = (dimLeft + blockSize.x - 1)/blockSize.x;
	int by = (dimRight + blockSize.y - 1)/blockSize.y;
	dim3 gridSize = dim3(bx, by);
	//matMulGPUKernel<<<gridSize, blockSize>>>(left, right, result, dimLeft, dimRight, dimCenter);
	//cudaDeviceSynchronize();

	Matrix<T> leftCPU = left.toDevice(0);
	Matrix<T> rightCPU = right.toDevice(0);
	
	return matMulSeq(leftCPU, rightCPU).toDevice(1);
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
