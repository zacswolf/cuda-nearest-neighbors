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

	// newData = trainData x rpMat, numDataPoints by newDim
	Matrix<T> newData = Matrix<T>::matMulSeq(trainData, rpMat);

	// newPredict = testData x rpMat, numDataPoints by newDim
	Matrix<T> newPredict = Matrix<T>::matMulSeq(testData, rpMat);

	return seqNormal(newData, trainLabels, newPredict);
}

template <typename T, typename G>
G* seqJLBernoulli(Matrix<T> &trainData, Matrix<G> &trainLabels, Matrix<T> &testData, int newDim) {
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
G* seqJLFast(Matrix<T> &trainData, Matrix<G> &trainLabels, Matrix<T> &testData, int newDim) {
		throw NotImplementedException("SEQUENTIAL::JLFAST");
}
