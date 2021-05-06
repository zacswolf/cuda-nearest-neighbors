#include <iostream>
#include <unistd.h>
#include <fstream>
#include <chrono>

#include "utils.h"
#include "matrix.h"
#include "csv_data_loader.h"
#include "nearest_neighbor_sequential.h"
#include "nearest_neighbor_gpu.h"
#include "exceptions.h"

enum class Mode {NORMAL, JLGAUSSIAN, JLBERNOULLI, JLFAST}; 

template <typename T, typename G>
G* nearestNeighbor(Mode mode, bool gpu, Matrix<T> &trainData, Matrix<G> &trainLabels, Matrix<T> &testData, int newDim) {
	switch(mode) {
		case Mode::NORMAL:
			if (gpu) {
				return gpuNormal(trainData, trainLabels, testData);
			} else {
				return seqNormal(trainData, trainLabels, testData);
			}
			//break;
		case Mode::JLGAUSSIAN:
			if (gpu) {
				return gpuJLGaussian(trainData, trainLabels, testData, newDim);
			} else {
				return seqJLGaussian(trainData, trainLabels, testData, newDim);
			}
			//break;
		case Mode::JLBERNOULLI:
			if (gpu) {
				return gpuJLBernoulli(trainData, trainLabels, testData, newDim);
			} else {
				return seqJLBernoulli(trainData, trainLabels, testData, newDim);
			}
			//break;
		case Mode::JLFAST:
			if (gpu) {
				return gpuJLFast(trainData, trainLabels, testData, newDim);
			} else {
				return seqJLFast(trainData, trainLabels, testData, newDim);
			}
			//break;
		default:
			throw NotImplementedException("The nearestNeighbor mode");
	}
	//return nullptr;
}


int main(int argc, char *argv[]) { 
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	// Arguments 
	std::string inputTrainDatasetPath;
	std::string inputTestDatasetPath;
	bool gpu = false;
	Mode mode = Mode::NORMAL;
	int newDim = -1;
	bool outputLabels = false;
	
	// Load Arguments
	int opt;
	while ((opt = getopt(argc, argv, "d:p:gm:n:l")) != -1) {
		switch (opt) {
		case 'd':
			// inputTrainDatasetPath: a string specifying the path to the input traindata file
			inputTrainDatasetPath = optarg;
			break;
		case 'p':
			// inputTestDatasetPath: a string specifying the path to the input testdata points file
			inputTestDatasetPath = optarg;
			break;
		case 'g':
			// gpu: a flag to enable the GPU implementations
			gpu = true;
			break;
		case 'm':
			// mode: an integer specifying the mode, look at Mode Enum
			mode = static_cast<Mode>(atoi(optarg));
			break;
		case 'n':
			// newDim; a int specifying the new dimention of the approximation algorithm
			// required for non Normal modes
			newDim = atoi(optarg);
		case 'l':
			// outputLabels: a flag to print the predicted test labels
			outputLabels = true;
			break;
		default:
			break;
		}
	}

	// Print Arguments
	#if DEBUG
		printf("inputTrainDatasetPath %s   inputTestDatasetPath %s   ", 
			   inputTrainDatasetPath.c_str(), 
			   inputTestDatasetPath.c_str());
		
		const char* gpuStr = (gpu==true)? "GPU": "SEQUENTIAL";
		switch (mode) {
		case Mode::NORMAL:
			printf("mode %s::NORMAL\n", gpuStr);
			break;
		case Mode::JLGAUSSIAN:
			printf("mode %s::JLGAUSSIAN\n", gpuStr);
			break;
		case Mode::JLBERNOULLI:
			printf("mode %s::JLBERNOULLI\n", gpuStr);
			break;
		case Mode::JLFAST:
			printf("mode %s::JLFAST\n", gpuStr);
			break;
		}
	#endif

	// Get train dataset
	CSVDataLoader<float> dl;
	Matrix<float> trainDataRaw = dl.load(inputTrainDatasetPath);

	// Split labels from train dataset
	pair<Matrix<float>, Matrix<float>> p1 = trainDataRaw.popColumn(-1); 
	Matrix<bool> trainLabels = p1.first.convert<bool>();
	Matrix<float> trainData = p1.second;

	#if DEBUG
		printf("trainData\n");
		trainData.print();
		printf("trainLabels\n");
		trainLabels.print();
	#endif

	// Get test dataset
	Matrix<float> testDataRaw = dl.load(inputTestDatasetPath);
	
	// Split labels from test dataset
	pair<Matrix<float>, Matrix<float>> p2 = testDataRaw.popColumn(-1); 
	Matrix<float> testLabels = p2.first;
	Matrix<float> testData = p2.second;

	#if DEBUG
		printf("testData\n");
		testData.print();
		printf("testLabels\n");
		testLabels.print();
	#endif

	//CLEANUP
	delete [] trainDataRaw.data;
	delete [] testDataRaw.data;

	// Check input file dimensions
	if (trainData.numCols != testData.numCols){
		throw std::invalid_argument("Data and PredictData dimentions are not the same");
	}

	// check new dim
	if (mode != Mode::NORMAL) {
		if (newDim == -1) {
			throw std::invalid_argument("Need to provide an newDim");
		}
		if ((0 >= newDim) || (newDim > trainData.numCols)) {
			throw std::invalid_argument("The value of newDim should be in (0, dim]");
		}
	}

	const bool *predictedTestLabels;
	if (gpu) {
		Matrix<float> d_trainData = trainData.toDevice(gpu);
		Matrix<bool> d_trainLabels = trainLabels.toDevice(gpu);
		Matrix<float> d_testData = testData.toDevice(gpu);

		// Call nearest neighbors
		printf("Calling nearestNeighbor\n");
		predictedTestLabels = nearestNeighbor(mode, gpu, d_trainData, d_trainLabels, d_testData, newDim);
	} else {
		// Call nearest neighbors
		printf("Calling nearestNeighbor\n");
		predictedTestLabels = nearestNeighbor(mode, gpu, trainData, trainLabels, testData, newDim);
	}

	printf("Finished nearestNeighbor\n");

	int numTestPoints = testData.numRows;

	int accuracySum = 0;
	for (int i = 0; i < numTestPoints ; i++) {
		accuracySum += predictedTestLabels[i]==testLabels.data[testLabels.index(i,0)];
	}

	float testAccuracy = static_cast<float>(accuracySum)/numTestPoints;
	printf("testAccuracy %.3f\n", testAccuracy);

	if (outputLabels) {
		if (predictedTestLabels != nullptr) {
			printf("predictedTestLabels\n[");
			for (int i = 0; i < numTestPoints; i++) {
				cout << predictedTestLabels[i] << " ";
			}
			printf("]\n");
		} else {
			printf("predictedTestLabels is null\n");
		}
	}
	
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	printf("End-to-end time: %f ms\n", elapsed);

	// final outputs for analytics program
	printf("%f, %f\n", testAccuracy, elapsed);

	return 0;
}
