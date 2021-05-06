#include "nearest_neighbor_sequential.h"
#include "vector"

template <typename T, typename G>
G* seqNormal(Matrix<T> &trainData, Matrix<G> &trainLabels, Matrix<T> &testData) {
	int numPredictPoints = testData.numRows;
	int numDataPoints = trainData.numRows;

	G *predictedLabels = new G(numPredictPoints);

	// Nearest neighbors
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

	return predictedLabels;
}

template <typename T, typename G>
G* seqJLGaussian(Matrix<T> &trainData, Matrix<G> &trainLabels, Matrix<T> &testData, int newDim) {
	int dim = trainData.numCols;

	// Make a random projection matrix of size dim x newDim
	Matrix<float> rpMat = Matrix<float>(dim, newDim);
	std::normal_distribution<float> distribution(0., 1.);
	rpMat.fill(distribution);

	// newTrainData = trainData x rpMat, numDataPoints by newDim
	Matrix<decltype(std::declval<T&>() * std::declval<G&>())> newTrainData = Matrix<T>::matMulSeq(trainData, rpMat);

	// newTestData = testData x rpMat, numDataPoints by newDim
	Matrix<decltype(std::declval<T&>() * std::declval<G&>())> newTestData = Matrix<T>::matMulSeq(testData, rpMat);

	return seqNormal(newTrainData, trainLabels, newTestData);
}

template <typename T, typename G>
G* seqJLBernoulli(Matrix<T> &trainData, Matrix<G> &trainLabels, Matrix<T> &testData, int newDim) {
	int dim = trainData.numCols;

	// Make a random projection matrix of size dim x newDim
	Matrix<int8_t> rpMat = Matrix<int8_t>(dim, newDim);
	std::bernoulli_distribution distribution(.5);
	rpMat.fill(distribution);

	// newTrainData = trainData x rpMat, numDataPoints by newDim

	Matrix<decltype(std::declval<T&>() * std::declval<G&>())> newTrainData = Matrix<T>::matMulSeq(trainData, rpMat);

	// newTestData = testData x rpMat, numDataPoints by newDim
	Matrix<decltype(std::declval<T&>() * std::declval<G&>())> newTestData = Matrix<T>::matMulSeq(testData, rpMat);

	return seqNormal(newTrainData, trainLabels, newTestData);
}


template <typename T>
Matrix<T> applySHD(Matrix<T> &mat, int newDim, Matrix<int8_t> &D, Matrix<int> &S) {
	int dim = mat.numCols;
	int numPoints = mat.numRows;
	
	assert(dim==D.numRows);
	assert(newDim==S.numCols);

	// Mult mat x D
	Matrix<decltype(std::declval<T&>() * std::declval<int8_t&>())> result1 = Matrix<T>::matMulDiagSeq(mat, D);

	// Mult result1 x H where H is a Walsh-Hadamard matrix.
	Matrix<decltype(std::declval<T&>() * std::declval<int8_t&>())> result2 = Matrix<decltype(std::declval<T&>() * std::declval<int8_t&>())>::matMulWalshHadamardSeq(result1);

	// Mult result2 x S 
	Matrix<decltype(std::declval<T&>() * std::declval<int8_t&>())> result3 = Matrix<decltype(std::declval<T&>() * std::declval<int8_t&>())>::matMulWithOneHotSeq(result2, S);

	//CLEANUP
	delete [] result1.data;
	delete [] result2.data;

	return result3;
}

template <typename T, typename G>
G* seqJLFast(Matrix<T> &trainData, Matrix<G> &trainLabels, Matrix<T> &testData, int newDim) {
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

	// apply SHD
	Matrix<T> newTrainData = applySHD(trainData, newDim, D, S);
	Matrix<T> newTestData = applySHD(testData, newDim, D, S);

	//CLEANUP
	delete [] S.data;
	delete [] D.data;
	
	return seqNormal(newTrainData, trainLabels, newTestData);
}
