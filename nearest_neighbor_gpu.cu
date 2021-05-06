#include "nearest_neighbor_gpu.h"
#include "vector"
#include "math.h"
//#include "cuda_matrix.h"

template <typename T, typename G>
__global__ void gpuNormalKernel1D(Matrix<T> d_trainData, Matrix<G> d_trainLabels, Matrix<T> d_testData, double closestDistance, G *d_predictedLabels, int numDataPoints) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // test point index

	int closestPoint = 0;
	
	for (int j = 0; j < numDataPoints; j++) {
		double currentDistance = Matrix<T>::l2RowDistanceSeq(d_trainData, j, d_testData, i);
		// Save if currentDistance < closestDistance
		if (currentDistance < closestDistance) {
			closestPoint = j;
			closestDistance = currentDistance;
			d_predictedLabels[i] = d_trainLabels.data[j];
		}
	}
}

template <typename T, typename G>
G* gpuNormal(Matrix<T> &trainData, Matrix<G> &trainLabels, Matrix<T> &testData) {
	int numPredictPoints = testData.numRows;
	int numDataPoints = trainData.numRows;
	double closestDistance = std::numeric_limits<double>::max();

	G *d_predictedLabels;
	cudaMalloc(&d_predictedLabels, numPredictPoints * sizeof(G));

	printf("Running on GPU with %d predict points and %d data points \n", numPredictPoints, numDataPoints);
	
	int blockSize = (int)ceil(((float)numPredictPoints/512.0f));

	gpuNormalKernel1D<<<blockSize, 512>>>(trainData, trainLabels, testData, closestDistance, d_predictedLabels, numDataPoints);
	cudaDeviceSynchronize();

	G *predictedLabels = (G *)malloc(numPredictPoints * sizeof(G));
	cudaMemcpy(predictedLabels, d_predictedLabels, numPredictPoints * sizeof(G), cudaMemcpyDeviceToHost);

	cudaFree(d_predictedLabels);

	return predictedLabels;
}
template bool* gpuNormal<float, bool>(Matrix<float>&, Matrix<bool>&, Matrix<float>&);

template <typename T, typename G>
G* gpuJLGaussian(Matrix<T> &trainData, Matrix<G> &trainLabels, Matrix<T> &testData, int newDim) {
	int dim = trainData.numCols;
	
	// Make a random projection matrix of size dim x newDim
	Matrix<float> rpMat = Matrix<float>(dim, newDim);
	std::normal_distribution<float> distribution(0., 1.);
	rpMat.fill(distribution);
	Matrix<float> d_rpMat = rpMat.toDevice(trainData.device);

	// newTrainData = trainData x rpMat, numDataPoints by newDim
	Matrix<decltype(std::declval<T&>() * std::declval<G&>())> newTrainData = Matrix<T>::matMulGPU(trainData, d_rpMat);

	// newTestData = testData x rpMat, numDataPoints by newDim
	Matrix<decltype(std::declval<T&>() * std::declval<G&>())> newTestData = Matrix<T>::matMulGPU(testData, d_rpMat);

	return gpuNormal(newTrainData, trainLabels, newTestData);
}
template bool* gpuJLGaussian<float, bool>(Matrix<float>&, Matrix<bool>&, Matrix<float>&, int);

template <typename T, typename G>
G* gpuJLBernoulli(Matrix<T> &trainData, Matrix<G> &trainLabels, Matrix<T> &testData, int newDim) {
	int dim = trainData.numCols;

	// Make a random projection matrix of size dim x newDim
	Matrix<bool> rpMat = Matrix<bool>(dim, newDim);
	std::bernoulli_distribution distribution(.5);
	rpMat.fill(distribution);
	Matrix<bool> d_rpMat = rpMat.toDevice(trainData.device);

	// newTrainData = trainData x rpMat, numDataPoints by newDim
	Matrix<decltype(std::declval<T&>() * std::declval<G&>())> newTrainData = Matrix<T>::matMulGPU(trainData, d_rpMat);

	// newTestData = testData x rpMat, numDataPoints by newDim
	Matrix<decltype(std::declval<T&>() * std::declval<G&>())> newTestData = Matrix<T>::matMulGPU(testData, d_rpMat);

	return gpuNormal(newTrainData, trainLabels, newTestData);
}
template bool* gpuJLBernoulli<float, bool>(Matrix<float>&, Matrix<bool>&, Matrix<float>&, int);


template <typename T>
Matrix<T> applySHDGPU(Matrix<T> &mat, int newDim, Matrix<int8_t> &D, Matrix<int> &S) {
	int dim = mat.numCols;
	int numPoints = mat.numRows;
	
	assert(dim==D.numRows);
	assert(newDim==S.numCols);

	// Mult mat x D
	Matrix<decltype(std::declval<T&>() * std::declval<int8_t&>())> result1 = Matrix<T>::matMulDiagGPU(mat, D);

	// Mult result1 x H where H is a Walsh-Hadamard matrix.
	Matrix<decltype(std::declval<T&>() * std::declval<int8_t&>())> result2 = Matrix<decltype(std::declval<T&>() * std::declval<int8_t&>())>::matMulWalshHadamardGPU(result1);

	// Mult result2 x S 
	Matrix<decltype(std::declval<T&>() * std::declval<int8_t&>())> result3 = Matrix<decltype(std::declval<T&>() * std::declval<int8_t&>())>::matMulWithOneHotGPU(result2, S);

	return result3;
}

template <typename T, typename G>
G* gpuJLFast(Matrix<T> &trainData, Matrix<G> &trainLabels, Matrix<T> &testData, int newDim) {
	assert(trainData.numCols == testData.numCols);
	int dim = trainData.numCols;

	// SHD is a newDim x dim matrix

	// scalar for SHD
	// float scalar = sqrt(static_cast<float>(dim)/newDim);

	// make S, a vector of indices to represent the columns of a matrix with one-hot cols
	Matrix<int> S = Matrix<int>(1, newDim);

	std::uniform_int_distribution<> distribution(0, dim-1);
	S.fill(distribution);

	// make, D a Rademacher vector (unif +- 1) representing a diagonal matrix
	Matrix<int8_t> D = Matrix<int8_t>(dim, 1);

	std::bernoulli_distribution distribution2(.5);
	D.fill(distribution2);

	Matrix<int> d_S = S.toDevice(trainData.device);
	Matrix<int8_t> d_D = D.toDevice(trainData.device);

	// apply SHD
	Matrix<T> newTrainData = applySHDGPU(trainData, newDim, d_D, d_S);
	Matrix<T> newTestData = applySHDGPU(testData, newDim, d_D, d_S);

	//CLEANUP
	delete [] S.data;
	delete [] D.data;
	
	return gpuNormal(newTrainData, trainLabels, newTestData);
}
template bool* gpuJLFast<float, bool>(Matrix<float>&, Matrix<bool>&, Matrix<float>&, int);