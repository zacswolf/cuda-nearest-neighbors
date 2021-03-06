#include "matrix.h"

template <typename T>
__host__ void Matrix<T>::fill(std::normal_distribution<float> distribution) {
	std::default_random_engine generator(0);

	for (int i = 0; i < (this->numRows * this->numCols); i++) {
		this->data[i] = distribution(generator);
	}
}

template <typename T>
__host__ void Matrix<T>::fill(std::bernoulli_distribution distribution) {
	std::default_random_engine generator(0);

	for (int i = 0; i < (this->numRows * this->numCols); i++) {
		this->data[i] = distribution(generator)*2 - 1;
	}
}

template <typename T>
__host__ void Matrix<T>::fill(std::uniform_int_distribution<> distribution) {
	std::default_random_engine generator(0);

	for (int i = 0; i < (this->numRows * this->numCols); i++) {
		this->data[i] = distribution(generator);
	}
}

template <typename T>
__host__ __device__ void Matrix<T>::fill(T val) {
	if (this->device==0){
		for (int i = 0; i < (this->numRows * this->numCols); i++) {
			this->data[i] = val;
		}
	} else {
		cudaMemset(this->data, val, (this->numRows * this->numCols)*sizeof(T));
	}
}

template <typename T>
Matrix<T> Matrix<T>::transpose() {
	// int temp = this->numRows;
	// this->numRows = this->numCols;
	// this->numCols = temp;


	// T* data = new T[this->numCols* this->numRows];
	Matrix<T> thisT = Matrix<T>(this->numCols, this->numRows);

	for (int r = 0; r < this->numRows; r++) {
		for (int c = 0; c < this->numCols; c++) {
			thisT.data[thisT.index(c,r)] = this->data[this->index(r,c)];
		}
	}

	return thisT;
}

template <typename T>
__host__ __device__ int Matrix<T>::index(int row, int col) {
	return row*(this->numCols) + col;
}

// Move matrix between CPU and device
template <typename T>
__host__ Matrix<T> Matrix<T>::toDevice(int device) {
	if (this->device == 0 && device != 0) {
		// assert(device != this->device);


		int dataBytes = (this->numRows * this->numCols) * sizeof(T);

		T *dataRaw;
		cudaMalloc(&dataRaw, dataBytes);
		cudaMemcpy(dataRaw, this->data, dataBytes, cudaMemcpyHostToDevice);


		Matrix<T> ret = Matrix<T>(dataRaw, this->numRows, this->numCols, device);

		return ret;
	} else if (this->device != 0 && device == 0) {
		// Move back to CPU
		// assert(device != this->device);

		int dataBytes = (this->numRows * this->numCols) * sizeof(T);

		T *dataRaw = new T[this->numRows * this->numCols];
		cudaMemcpy(dataRaw, this->data, dataBytes, cudaMemcpyDeviceToHost);

		Matrix<T> ret = Matrix<T>(dataRaw, this->numRows, this->numCols, device);
		return ret;
	} else {
		throw NotImplementedException("Matrix<T>::toDevice()");
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
	int dimLeft = left.numRows;
	int dimCenter = left.numCols;
	int dimRight = right.numCols;
	assert(dimCenter == right.numRows);

	Matrix result = Matrix<decltype(std::declval<T&>() * std::declval<G&>())>(dimLeft, dimRight);
	result.fill(0);

	// Matrix Mult
    for (int i = 0; i < dimLeft; i++) {
        for (int j = 0; j < dimRight; j++) {
            for (int k = 0; k < dimCenter; k++) {
                result.data[result.index(i, j)] += left.data[left.index(i, k)] * right.data[right.index(k, j)];
			}
        }
    }
	
	return result;
}

#define TILE_WIDTH 32

template <typename T, typename G>
__global__ void matMulGPUKernel2DShmem(Matrix<T> left, Matrix<G> right, Matrix<decltype(std::declval<T&>() * std::declval<G&>())> result, int dimLeft, int dimRight, int dimCenter) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ T leftCache[TILE_WIDTH][TILE_WIDTH];
	__shared__ T rightCache[TILE_WIDTH][TILE_WIDTH];

	decltype(std::declval<T&>() * std::declval<G&>()) matmulValue = 0;
	for (int m = 0; m < (TILE_WIDTH + dimLeft - 1)/TILE_WIDTH; m++) {
		leftCache[threadIdx.x][threadIdx.y] = left.data[left.index(i, (m * TILE_WIDTH + threadIdx.y))];
		rightCache[threadIdx.x][threadIdx.y] = right.data[right.index((m * TILE_WIDTH + threadIdx.x), j)];
		__syncthreads();

		for (int k = 0; k < TILE_WIDTH; k++) {
			matmulValue += leftCache[threadIdx.x][k] * rightCache[k][threadIdx.y];
		}
	}

	//printf("SHMEM Matmul value: %f\n", matmulValue);
	//printf("dimleft: %d, Block idx: %d\n", dimLeft, blockIdx.x);

	result.data[result.index(i, j)] = matmulValue;
}

template <typename T, typename G>
__global__ void matMulGPUKernel2D(Matrix<T> left, Matrix<G> right, Matrix<decltype(std::declval<T&>() * std::declval<G&>())> result, int dimLeft, int dimRight, int dimCenter) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	decltype(std::declval<T&>() * std::declval<G&>()) matmulValue = 0;
	for (int k = 0; k < dimCenter; k++) {
		matmulValue += left.data[left.index(i, k)] * right.data[right.index(k, j)];
	}

	result.data[result.index(i, j)] = matmulValue;
}

template <typename T>
template <typename G>
__host__ Matrix<decltype(std::declval<T&>() * std::declval<G&>())> Matrix<T>::matMulGPU(Matrix<T> &left, Matrix<G> &right) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int dimLeft = left.numRows;
	int dimCenter = left.numCols;
	int dimRight = right.numCols;
	assert(dimCenter == right.numRows);

	printf("Matmul with %d x %d matrix and %d x %d matrix\n", dimLeft, dimCenter, right.numRows, dimRight);

	assert(left.device == right.device);
	assert(left.device != 0);

	Matrix result = Matrix<decltype(std::declval<T&>() * std::declval<G&>())>(dimLeft, dimRight).toDevice(left.device); // TODO: improve this
	result.fill(0);

	// Launching a 2D kernel
	int xBlock = (int)ceil(((float)dimLeft/512.0f));
	int yBlock = (int)ceil(((float)dimRight/512.0f));
	dim3 blockSize(xBlock, yBlock);
	int bx = (dimLeft + blockSize.x - 1)/blockSize.x;
	int by = (dimRight + blockSize.y - 1)/blockSize.y;
	dim3 gridSize = dim3(bx, by);
	cudaEventRecord(start);
	matMulGPUKernel2D<<<gridSize, blockSize>>>(left, right, result, dimLeft, dimRight, dimCenter);
	cudaEventRecord(stop);
	/*
	//int blockDim = 32;
	dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
	int xGrid = (int)ceil(((float)dimLeft/(float)TILE_WIDTH));
	int yGrid = (int)ceil(((float)dimRight/(float)TILE_WIDTH));
	dim3 gridSize(xGrid, yGrid);
	cudaEventRecord(start);
	matMulGPUKernel2DShmem<<<gridSize, blockSize>>>(left, right, result, dimLeft, dimRight, dimCenter);
	cudaEventRecord(stop);
	*/

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU matmul took %f ms\n", milliseconds);

	return result;
}

template <typename T, typename G>
__global__ void matMulDiagGPUKernel(Matrix<T> left, Matrix<G> diag, Matrix<decltype(std::declval<T&>() * std::declval<G&>())> result, int dimCenter) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int k = 0; k < dimCenter; k++) {
		result.data[result.index(i, k)] = diag.data[k] * left.data[left.index(i, k)];
	}
}

template <typename T>
template <typename G>
__host__ Matrix<decltype(std::declval<T&>() * std::declval<G&>())> Matrix<T>::matMulDiagGPU(Matrix<T> &left, Matrix<G> &diag) {
	int dimLeft = left.numRows;
	int dimCenter = left.numCols;
	assert(dimCenter == diag.numRows);
	assert(diag.numCols == 1);

	// Mult mat x D
	Matrix<decltype(std::declval<T&>() * std::declval<G&>())> result = Matrix<T>(dimLeft, dimCenter).toDevice(left.device);

	int blockSize = (int)ceil(((float)dimLeft/512.0f));
	matMulDiagGPUKernel<<<blockSize, 512>>>(left, diag, result, dimCenter);
	cudaDeviceSynchronize();

	return result;
}

template <typename T>
__global__ void matMulWalshHadamardGPUKernel(Matrix<T> left, Matrix<T> result, int dimLeft, int dimCenter) {
	int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;

	int log2dim = ceil(log2(dimCenter));
	int hShape = pow(2,log2dim);

	int order = 1;
	int stride = 2;
	int split = stride/2;

	Matrix<T> mats [] = {Matrix<T>(hShape, 1), Matrix<T>(hShape, 1)};
	mats[0].fill(0);
	mats[1].fill(0);

	int newIdx = 0;

	for (int i = 0; i < dimCenter; i++) {
		mats[newIdx].data[i] = left.data[left.index(pointIdx, i)];
	}


	for (order = 2; order < log2dim; order++) { // cant parallize
		newIdx = !newIdx;

		stride = pow(2, order);
		split = stride/2;

		for (int strideId = 0; strideId < hShape/stride; strideId++) {
			for (int idx = 0; idx < split; idx++) { 
				// c0
				mats[newIdx].data[strideId*stride+idx] = mats[!newIdx].data[strideId*stride+idx] + mats[!newIdx].data[strideId*stride+idx+(split/2)];

				// c1
				mats[newIdx].data[strideId*stride+idx+split] = mats[!newIdx].data[strideId*stride+idx+split] - mats[!newIdx].data[strideId*stride+idx+split+(split/2)];
			}
		}
	}

	for (int d = 0; d < dimCenter; d++) {
		result.data[result.index(pointIdx, d)] = mats[newIdx].data[d];
	}

	//CLEANUP
	delete [] mats[0].data;
	delete [] mats[1].data;
}

template <typename T>
__host__ Matrix<T> Matrix<T>::matMulWalshHadamardGPU(Matrix<T> left) {
	int dimLeft = left.numRows;
	int dimCenter = left.numCols;
	
	Matrix<T> result = Matrix<T>(dimLeft, dimCenter).toDevice(left.device);

	assert(dimCenter > 1); // TODO support this
	
	int blockSize = (int)ceil(((float)dimLeft/512.0f));
	matMulWalshHadamardGPUKernel<<<blockSize, 512>>>(left, result, dimLeft, dimCenter);
	cudaDeviceSynchronize();

	return result;
}

template <typename T, typename G>
__global__ void matMulWithOneHotGPUKernel(Matrix<T> left, Matrix<G> oneHot, Matrix<T> result, int dimRight) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	for(int j = 0; j < dimRight; j++) {
		int onehotdim = oneHot.data[j];
		result.data[result.index(i, j)] = left.data[left.index(i, onehotdim)];
	}
}

template <typename T>
template <typename G>
__host__ Matrix<decltype(std::declval<T&>() * std::declval<G&>())> Matrix<T>::matMulWithOneHotGPU(Matrix<T> left, Matrix<G> oneHot) {
	int dimLeft = left.numRows;
	int dimCenter = left.numCols;
	int dimRight = oneHot.numCols;

	assert(oneHot.numRows == 1);

	Matrix<T> result = Matrix<T>(dimLeft, dimRight).toDevice(left.device);

	int blockSize = (int)ceil(((float)dimLeft/512.0f));
	matMulWithOneHotGPUKernel<<<blockSize, 512>>>(left, oneHot, result, dimRight);
	cudaDeviceSynchronize();
	
	return result;
}

template <typename T>
template <typename G>
__host__ Matrix<decltype(std::declval<T&>() * std::declval<G&>())> Matrix<T>::matMulDiagSeq(Matrix<T> &left, Matrix<G> &diag) {
	int dimLeft = left.numRows;
	int dimCenter = left.numCols;
	assert(dimCenter == diag.numRows);
	assert(diag.numCols == 1);

	// Mult mat x D
	Matrix<decltype(std::declval<T&>() * std::declval<G&>())> result = Matrix<T>(dimLeft, dimCenter);

	for (int i = 0; i < dimLeft; i++) {
		for (int k = 0; k < dimCenter; k++) {
			result.data[result.index(i, k)] = diag.data[k] * left.data[left.index(i, k)];
		}
	}

	return result;
}

template <typename T>
__host__ Matrix<T> Matrix<T>::matMulWalshHadamardSeq(Matrix<T> left) {
	int dimLeft = left.numRows;
	int dimCenter = left.numCols;
	
	Matrix<T> result = Matrix<T>(dimLeft, dimCenter);

	assert(dimCenter > 1); // TODO support this
	
	int log2dim = ceil(log2(dimCenter));
	int hShape = pow(2,log2dim);
	
	for (int pointIdx=0; pointIdx < dimLeft; pointIdx++) {  // should parallize
		int order = 1;
		int stride = 2;
		int split = stride/2;

		Matrix<T> mats [] = {Matrix<T>(hShape, 1), Matrix<T>(hShape, 1)};
		mats[0].fill(0);
		mats[1].fill(0);

		int newIdx = 0;

		for (int i = 0; i < dimCenter; i++) {
			mats[newIdx].data[i] = left.data[left.index(pointIdx, i)];
		}


		for (order = 2; order < log2dim; order++) { // cant parallize
			newIdx = !newIdx;

			stride = pow(2, order);
			split = stride/2;

			for (int strideId = 0; strideId < hShape/stride; strideId++) { // could parallize, these two loops combined are of over original dim
				for (int idx = 0; idx < split; idx++) {  // could parallize
					// c0
					mats[newIdx].data[strideId*stride+idx] = mats[!newIdx].data[strideId*stride+idx] + mats[!newIdx].data[strideId*stride+idx+(split/2)];

					// c1
					mats[newIdx].data[strideId*stride+idx+split] = mats[!newIdx].data[strideId*stride+idx+split] - mats[!newIdx].data[strideId*stride+idx+split+(split/2)];
				}
			}

			// idk why this doesnt work
			// for (int idx = 0; idx < hShape; idx++) { // could parallize, these two loops combined are of over original dim
			// 	mats[newIdx].data[idx] = mats[!newIdx].data[idx] + mats[!newIdx].data[idx + (split/2)];
			// }

		}

		for (int d = 0; d < dimCenter; d++) {
			result.data[result.index(pointIdx, d)] = mats[newIdx].data[d];
		}

		//CLEANUP
		delete [] mats[0].data;
		delete [] mats[1].data;
	} 

	return result;
}

template <typename T>
template <typename G>
__host__ Matrix<decltype(std::declval<T&>() * std::declval<G&>())> Matrix<T>::matMulWithOneHotSeq(Matrix<T> left, Matrix<G> oneHot) {
	int dimLeft = left.numRows;
	int dimCenter = left.numCols;
	int dimRight = oneHot.numCols;

	assert(oneHot.numRows == 1);

	Matrix<T> result = Matrix<T>(dimLeft, dimRight);

	for (int i = 0; i < dimLeft; i++) { // should parallize
		for(int j = 0; j < dimRight; j++) {
			int onehotdim = oneHot.data[j];
			result.data[result.index(i, j)] = left.data[left.index(i, onehotdim)];
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
__host__ __device__ void Matrix<T>::print() {
	if (this->numCols != 1) {
		printf("[\n");
		for (int row = 0; row < this->numRows; row++) {
			printf("[ ");
			for (int col = 0; col < this->numCols; col++) {
				printf("%s ", std::to_string(this->data[this->index(row, col)]).c_str());
				// cout << this->data[this->index(row, col)] << " ";
			}
			printf("]\n");
		}
		printf("]\n");
	} else {
		printf("[");
		for (int row = 0; row < this->numRows; row++) {
			// cout << this->data[this->index(row, 0)] << " ";
			printf("%s ", std::to_string(this->data[this->index(row, 0)]).c_str());
		}
		printf("]\n");
	}
}

// template class Matrix<float>;
// template class Matrix<bool>;
// template class Matrix<int>;
