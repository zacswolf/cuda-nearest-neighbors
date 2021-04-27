#include "nearest_neighbor_sequential.h"

float* seqNormal(Matrix data, Matrix labels, Matrix predictData) {
	int numPredictPoints = predictData.getNumRows();
	int numDataPoints = data.getNumRows();
	int dim = data.getNumCols();

	float *predictedLabels = new float(numPredictPoints);

	// Nearest neighbors
	for (int currentPredictPoint = 0; currentPredictPoint < numPredictPoints; currentPredictPoint++) {
		int closestPoint = 0;
		float closestDistance = std::numeric_limits<float>::max();
		for (int currentDataPoint = 0; currentDataPoint < numDataPoints; currentDataPoint++) {
			float currentDistance = 0;
			// l2 norm
			for (int d = 0; d < dim; d++) {
				int term = data.get(currentDataPoint, d) - predictData.get(currentPredictPoint, d);
				currentDistance += term*term;
			}
			currentDistance = sqrt(currentDistance);

			// Save if currentDistance < closestDistance
			bool newClosest = (currentDistance < closestDistance);
			closestPoint = (!newClosest)*closestPoint + newClosest*currentDataPoint;
			closestDistance = (!newClosest)*closestDistance + newClosest*currentDistance;
		}
		predictedLabels[currentPredictPoint] = labels.get(closestPoint, 0);
	}
	return predictedLabels;
}
