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

template <typename T, typename G>
G* gpuJLFast(Matrix<T> &trainData, Matrix<G> &trainLabels, Matrix<T> &testData, int newDim) {
		throw NotImplementedException("GPU::JLFAST");
}
template bool* gpuJLFast<float, bool>(Matrix<float>&, Matrix<bool>&, Matrix<float>&, int);