#include "nearest_neighbor_gpu.h"
#include "vector"
#include "math.h"
//#include "cuda_matrix.h"

template <typename T, typename G>
__global__ void gpuNormalKernel(Matrix<T> *d_trainData, Matrix<G> *d_trainLabels, Matrix<T> *d_testData, int *d_closestPoint, double *d_closestDistance, G *d_predictedLabels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // test point index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // train point index

	//printf("i: %d, j: %d\n", i, j);

	int closestPoint = d_closestPoint[i];
	double closestDistance = d_closestDistance[i];
	//printf("before distance: %d\n", (*d_trainData).data[0]);
	double currentDistance = Matrix<T>::l2RowDistanceSeq(*d_trainData, j, *d_testData, i);

	// Save if currentDistance < closestDistance
	if (currentDistance < closestDistance) {
		__syncthreads();
		d_closestPoint[i] = j;
		d_closestDistance[i] = currentDistance;
		//printf("distance: %f\n", currentDistance);
		//printf("closest point: %f\n", d_trainLabels->data[5]);
		d_predictedLabels[i] = d_trainLabels->data[j];
		__syncthreads();
		//printf("d_predictlabels[%d] = %f\n", d_trainLabels->data[closestPoint]);
		//printf("d_predictlabels[%d] = %d\n", i, d_trainLabels->data[closestPoint]);
	}
}

template <typename T, typename G>
G* gpuNormal(Matrix<T> &trainData, Matrix<G> &trainLabels, Matrix<T> &testData) {
	int numPredictPoints = testData.numRows;
	int numDataPoints = trainData.numRows;

	G *predictedLabels = new G(numPredictPoints);
	G *d_predictedLabels;
	cudaMalloc(&d_predictedLabels, numPredictPoints * sizeof(G));

	//Matrix<T>(arr, numPoints, dim);
	Matrix<T> *d_trainData, *d_testData;
	int trainDataBytes = (trainData.numRows * trainData.numCols) * sizeof(T);
	int testDataBytes = (testData.numRows * testData.numCols) * sizeof(T);
	//printf("Train data bytes: %d\n", trainDataBytes);
	cudaMalloc(&d_trainData, sizeof(Matrix<T>));
	cudaMalloc(&d_testData, sizeof(Matrix<T>));
	cudaMemcpy(d_trainData, &trainData, sizeof(Matrix<T>), cudaMemcpyHostToDevice);
	cudaMemcpy(d_testData, &testData, sizeof(Matrix<T>), cudaMemcpyHostToDevice);
	// Copy over data as well
	T *trainDataRaw, *testDataRaw;
	cudaMalloc(&trainDataRaw, trainDataBytes);
	cudaMalloc(&testDataRaw, testDataBytes);
	cudaMemcpy(trainDataRaw, trainData.data, trainDataBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(testDataRaw, testData.data, testDataBytes, cudaMemcpyHostToDevice);
	// Set device data pointers
	cudaMemcpy((void *)&(d_trainData->data), &trainDataRaw, sizeof(T *), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(d_testData->data), &testDataRaw, sizeof(T *), cudaMemcpyHostToDevice);

	Matrix<G> *d_trainLabels;
	int trainLabelsBytes = (trainLabels.numRows * trainLabels.numCols) * sizeof(G);
	cudaMalloc(&d_trainLabels, sizeof(Matrix<G>));
	cudaMemcpy(d_trainLabels, &trainLabels, sizeof(Matrix<G>), cudaMemcpyHostToDevice);
	G *trainLabelsRaw;
	cudaMalloc(&trainLabelsRaw, trainLabelsBytes);
	cudaMemcpy(trainLabelsRaw, trainLabels.data, trainLabelsBytes, cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(d_trainLabels->data), &trainLabelsRaw, sizeof(G *), cudaMemcpyHostToDevice);

	int *closestPoint = (int *)calloc(numPredictPoints, sizeof(int));
	int *d_closestPoint;
	cudaMalloc(&d_closestPoint, numPredictPoints * sizeof(int));
	double *closestDistance = (double *)malloc(numPredictPoints * sizeof(double));
	for (int i = 0; i < numPredictPoints; i++) { closestDistance[i] = std::numeric_limits<double>::max(); }
	double *d_closestDistance;
	cudaMalloc(&d_closestDistance, numPredictPoints * sizeof(double));

	cudaMemcpy(d_closestPoint, closestPoint, numPredictPoints * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_closestDistance, closestDistance, numPredictPoints * sizeof(double), cudaMemcpyHostToDevice);

	printf("Running on GPU with %d predict points and %d data points \n", numPredictPoints, numDataPoints);
	int xBlock = (int)ceil(((float)numPredictPoints/512.0f));
	int yBlock = (int)ceil(((float)numDataPoints/512.0f));
	printf("block size should be: %d %d\n", xBlock, yBlock);
	dim3 blockSize(xBlock, yBlock);
	int bx = (numPredictPoints + blockSize.x - 1)/blockSize.x;
	int by = (numDataPoints + blockSize.y - 1)/blockSize.y;
	dim3 gridSize = dim3(bx, by);
	gpuNormalKernel<<<gridSize, blockSize>>>(d_trainData, d_trainLabels, d_testData, d_closestPoint, d_closestDistance, d_predictedLabels);
	cudaDeviceSynchronize();

	cudaMemcpy(predictedLabels, d_predictedLabels, numPredictPoints * sizeof(G), cudaMemcpyDeviceToHost);

	// Nearest neighbors
	/*
	for (int currentTestPoint = 0; currentTestPoint < numPredictPoints; currentTestPoint++) {
		int closestPoint = 0;
		double closestDistance = std::numeric_limits<double>::max();
		for (int currentTrainPoint = 0; currentTrainPoint < numDataPoints; currentTrainPoint++) {
			// l2 distance squared
			double currentDistance = Matrix<T>::l2RowDistanceSeq(trainData, currentTrainPoint, testData, currentTestPoint);

			// Save if currentDistance < closestDistance
			bool newClosest = (currentDistance < closestDistance);

			closestPoint = (!newClosest)*closestPoint + newClosest*currentTrainPoint;
			closestDistance = (!newClosest)*closestDistance + newClosest*currentDistance;
		}
		predictedLabels[currentTestPoint] = trainLabels.data[closestPoint];
	}
	*/

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

	// newData = trainData x rpMat, numDataPoints by newDim
	Matrix<T> newData = Matrix<T>::matMulSeq(trainData, rpMat);

	// newPredict = testData x rpMat, numDataPoints by newDim
	Matrix<T> newPredict = Matrix<T>::matMulSeq(testData, rpMat);

	return seqNormal(newData, trainLabels, newPredict);
}

template <typename T, typename G>
G* gpuJLBernoulli(Matrix<T> &trainData, Matrix<G> &trainLabels, Matrix<T> &testData, int newDim) {
	int dim = trainData.numCols;

	// Make a random projection matrix of size dim x newDim
	Matrix<bool> rpMat = Matrix<bool>(dim, newDim);
	std::bernoulli_distribution distribution(.5);
	rpMat.fill(distribution);

	// newData = trainData x rpMat, numDataPoints by newDim

	Matrix<decltype(std::declval<T&>() * std::declval<G&>())> newData = Matrix<T>::matMulSeq(trainData, rpMat);

	// newPredict = testData x rpMat, numDataPoints by newDim
	Matrix<T> newPredict = Matrix<T>::matMulSeq(testData, rpMat);

	return seqNormal(newData, trainLabels, newPredict);
}

template <typename T, typename G>
G* gpuJLFast(Matrix<T> &trainData, Matrix<G> &trainLabels, Matrix<T> &testData, int newDim) {
		throw NotImplementedException("SEQUENTIAL::JLFAST");
}
