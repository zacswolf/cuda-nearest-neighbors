#include "nearest_neighbor_sequential.h"

float* seqNormal(Matrix *data, Matrix *labels, Matrix *predictData) {
	int numPredictPoints = predictData->getNumRows();
	int numDataPoints = data->getNumRows();
	int dim = data->getNumCols();

	float *predictedLabels = new float(numPredictPoints);

	// Nearest neighbors
	for (int currentPredictPoint = 0; currentPredictPoint < numPredictPoints; currentPredictPoint++) {
		int closestPoint = 0;
		float closestDistance = std::numeric_limits<float>::max();
		for (int currentDataPoint = 0; currentDataPoint < numDataPoints; currentDataPoint++) {
			float currentDistance = 0;
			// l2 norm
			for (int d = 0; d < dim; d++) {
				int term = data->get(currentDataPoint, d) - predictData->get(currentPredictPoint, d);
				currentDistance += term*term;
			}
			currentDistance = sqrt(currentDistance);

			// Save if currentDistance < closestDistance
			bool newClosest = (currentDistance < closestDistance);
			closestPoint = (!newClosest)*closestPoint + newClosest*currentDataPoint;
			closestDistance = (!newClosest)*closestDistance + newClosest*currentDistance;
		}
		predictedLabels[currentPredictPoint] = labels->get(closestPoint, 0);
	}
	return predictedLabels;
}

float* seqJLGaussian(Matrix *data, Matrix *labels, Matrix *predictData, float epsilon) {
	int numPredictPoints = predictData->getNumRows();
	int numDataPoints = data->getNumRows();
	int dim = data->getNumCols();

	// // TODO: find constant
	// int newDim = ceil(log(numDataPoints)/(epsilon*epsilon));

	// // Make a random projection matrix of size dim x newDim
	// std::default_random_engine generator;
	// std::normal_distribution<float> distribution(0., 1.);
	
	// vector<float> rpMatRaw(dim * newDim);

	// for (int i = 0; i < dim * newDim; i++){
	// 	rpMatRaw[i] = distribution(generator);
	// }

	// Matrix rpMat = Matrix(rpMatRaw, dim, newDim);

	// // newData = data x rpMat

	throw NotImplementedException("SEQUENTIAL::JLGAUSSIAN");

	float *predictedLabels = new float(numPredictPoints);
	return predictedLabels;

}
