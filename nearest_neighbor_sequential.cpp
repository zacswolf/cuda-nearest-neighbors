#include "nearest_neighbor_sequential.h"
#include "vector"
#define METHOD1 true

float* seqNormal(Matrix &trainData, Matrix &trainLabels, Matrix &testData) {
	int numPredictPoints = testData.numRows;
	int numDataPoints = trainData.numRows;

	float *predictedLabels = new float(numPredictPoints);

	// Nearest neighbors
	for (int currentTestPoint = 0; currentTestPoint < numPredictPoints; currentTestPoint++) {
		int closestPoint = 0;
		float closestDistance = std::numeric_limits<float>::max();
		for (int currentTrainPoint = 0; currentTrainPoint < numDataPoints; currentTrainPoint++) {
			// l2 distance squared
			float currentDistance = Matrix::l2RowDistanceSeq(trainData, currentTrainPoint, testData, currentTestPoint);

			// Save if currentDistance < closestDistance
			bool newClosest = (currentDistance < closestDistance);

			closestPoint = (!newClosest)*closestPoint + newClosest*currentTrainPoint;
			closestDistance = (!newClosest)*closestDistance + newClosest*currentDistance;
		}
		predictedLabels[currentTestPoint] = trainLabels.data[closestPoint];
	}

	return predictedLabels;
}

float* seqJLGaussian(Matrix &trainData, Matrix &trainLabels, Matrix &testData, int newDim) {
	int dim = trainData.numCols;

	// Constant is ~3
	// int newDim = ceil(3*log(numDataPoints)/(epsilon*epsilon));

	// Make a random projection matrix of size dim x newDim
	Matrix rpMat = Matrix(dim, newDim);
	std::normal_distribution<float> distribution(0., 1.);
	rpMat.fill(distribution);

	// newData = trainData x rpMat, numDataPoints by newDim
	Matrix newData = Matrix::matMulSeq(trainData, rpMat);

	// newPredict = testData x rpMat, numDataPoints by newDim
	Matrix newPredict = Matrix::matMulSeq(testData, rpMat);

	return seqNormal(newData, trainLabels, newPredict);
}

float* seqJLBernoulli(Matrix &trainData, Matrix &trainLabels, Matrix &testData, int newDim) {
	int dim = trainData.numCols;

	// Constant is ~3.5
	// int newDim = ceil(3.5*log(numDataPoints)/(epsilon*epsilon));

	// Make a random projection matrix of size dim x newDim
	Matrix rpMat = Matrix(dim, newDim);
	std::bernoulli_distribution distribution(.5);
	rpMat.fill(distribution);

	// newData = trainData x rpMat, numDataPoints by newDim
	Matrix newData = Matrix::matMulSeq(trainData, rpMat);

	// newPredict = testData x rpMat, numDataPoints by newDim
	Matrix newPredict = Matrix::matMulSeq(testData, rpMat);

	return seqNormal(newData, trainLabels, newPredict);
}

float* seqJLFast(Matrix &trainData, Matrix &trainLabels, Matrix &testData, int newDim) {
	throw NotImplementedException("SEQUENTIAL::JLFAST");
}
