#include "nearest_neighbor_sequential.h"
#include "vector"

float* seqNormal(Matrix &data, Matrix &labels, Matrix &predictData) {
	int numPredictPoints = predictData.numRows;
	int numDataPoints = data.numRows;
	int dim = data.numCols;

	float *predictedLabels = new float(numPredictPoints);

	// Nearest neighbors
	for (int currentPredictPoint = 0; currentPredictPoint < numPredictPoints; currentPredictPoint++) {
		int closestPoint = 0;
		float closestDistance = std::numeric_limits<float>::max();
		for (int currentDataPoint = 0; currentDataPoint < numDataPoints; currentDataPoint++) {
			float currentDistance = 0;
			// l2 norm
			for (int d = 0; d < dim; d++) {
				int term = data.data[index(data,currentDataPoint, d)] - predictData.data[index(predictData,currentPredictPoint, d)];
				currentDistance += term*term;
			}
			currentDistance = sqrt(currentDistance);

			// Save if currentDistance < closestDistance
			bool newClosest = (currentDistance < closestDistance);
			closestPoint = (!newClosest)*closestPoint + newClosest*currentDataPoint;
			closestDistance = (!newClosest)*closestDistance + newClosest*currentDistance;
		}
		predictedLabels[currentPredictPoint] = labels.data[index(labels,closestPoint, 0)];
	}
	return predictedLabels;
}

float* seqJLGaussian(Matrix &data, Matrix &labels, Matrix &predictData, float epsilon) {
	int numPredictPoints = predictData.numRows;
	int numDataPoints = data.numRows;
	int dim = data.numCols;

	// TODO: find constant
	int newDim = ceil(log(numDataPoints)/(epsilon*epsilon));

	// Make a random projection matrix of size dim x newDim
	Matrix rpMat = Matrix({vector<float>(dim*newDim), dim, newDim});
	std::normal_distribution<float> distribution(0., 1.);
	fill(rpMat, distribution);

	// newData = data x rpMat, numDataPoints by newDim
	Matrix newData = Matrix({vector<float>(numDataPoints*newDim), numDataPoints, newDim});

    for (int i = 0; i < numDataPoints; i++) {
        for (int j = 0; j < newDim; j++) {
            for (int k = 0; k < dim; k++) {
                // newData->set(i, j, newData->get(i,j) + data->get(i,k) * rpMat->get(k,j));
			}
        }
    }


	throw NotImplementedException("SEQUENTIAL::JLGAUSSIAN");

	float *predictedLabels = new float(numPredictPoints);
	return predictedLabels;

}
