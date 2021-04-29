#include <iostream>
#include <unistd.h>
#include <fstream>

#include "utils.h"
#include "matrix.h"
#include "csv_data_loader.h"
#include "nearest_neighbor_sequential.h"
#include "exceptions.h"

float* nearestNeighbor(Mode mode, bool gpu, Matrix &data, Matrix &labels, Matrix &predictData, float epsilon) {
	switch(mode) {
		case Mode::NORMAL:
			if (gpu) {
				throw NotImplementedException("GPU::NORMAL");
			} else {
				return seqNormal(data, labels, predictData);
			}
			break;
		case Mode::JLGAUSSIAN:
			if (gpu) {
				throw NotImplementedException("GPU::JLGAUSSIAN");
			} else {
				return seqJLGaussian(data, labels, predictData, epsilon);
			}
			break;
		case Mode::JLBERNOULLI:
			if (gpu) {
				throw NotImplementedException("GPU::JLBERNOULLI");
			} else {
				throw NotImplementedException("SEQUENTIAL::JLBERNOULLI");
			}
			break;
		case Mode::JLFAST:
			if (gpu) {
				throw NotImplementedException("GPU::JLFAST");
			} else {
				throw NotImplementedException("SEQUENTIAL::JLFAST");
			}
			break;
		default:
			throw NotImplementedException("The nearestNeighbor mode");
	}
	return nullptr;
}


int main(int argc, char *argv[]) { 
	// Arguments 
	std::string inputDatasetPath;
	std::string inputPredictDataPath;
	bool gpu = false;
	Mode mode = Mode::NORMAL;
	float epsilon = -1.;
	
	// Load Arguments
	int opt;
	while ((opt = getopt(argc, argv, "d:p:gm:e:")) != -1) {
		switch (opt) {
		case 'd':
			// inputDatasetPath: a string specifying the path to the input dataset file
			inputDatasetPath = optarg;
			break;
		case 'p':
			// inputPredictDataPath: a string specifying the path to the input predict points file
			inputPredictDataPath = optarg;
			break;
		case 'g':
			// gpu: a flag to enable the GPU implementations
			gpu = true;
			break;
		case 'm':
			// mode: an integer specifying the mode, look at Mode Enum
			mode = static_cast<Mode>(atoi(optarg));
			break;
		case 'e':
			// epsilon; a float specifying the accuracy of the approximation algorithm
			// required for non Normal modes
			epsilon = atof(optarg);
		default:
			break;
		}
	}

	// check epsilon
	if (mode != Mode::NORMAL) {
		if (epsilon == -1) {
			throw std::invalid_argument("Need to provide an epsilon");
		}
		if ((0. >= epsilon) || (epsilon >= 1.)) {
			throw std::invalid_argument("The value of epsilon should be in (0, 1)");
		}
	}

	// Print Arguments
	#if DEBUG
		printf("inputDatasetPath %s   inputPredictDataPath %s   ", 
			   inputDatasetPath.c_str(), 
			   inputPredictDataPath.c_str());
		
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
	

	// Get dataset
	CSVDataLoader dl;
	Matrix data = dl.loadFromFile(inputDatasetPath);

	// #if DEBUG
	//     data.print();
	// #endif

	// Split labels from dataset
	pair<Matrix, Matrix> p = popColumn(data, -1); 
	data = p.first;
	Matrix labels = p.second;

	#if DEBUG
		printf("data\n");
		print(data);
		printf("labels\n");
		print(labels);
	#endif

	// Get points to classify
	Matrix predictData = dl.loadFromFile(inputPredictDataPath);
	#if DEBUG
		printf("predictData\n");
		print(predictData);
	#endif

	// Check input file dimensions
	if (data.numCols != predictData.numCols){
		throw std::invalid_argument("Data and PredictData dimentions are not the same");
	}

	printf("Calling nearestNeighbor\n");

	// Call nearest neighbors
	float *predictedLabels = nearestNeighbor(mode, gpu, data, labels, predictData, epsilon);


	printf("Finished nearestNeighbor\n");

	int numPredictPoints = predictData.numRows;

	if (predictedLabels != nullptr) {
		printf("predictedLabels\n[\n");
		for (int i = 0; i < numPredictPoints; i++) {
			printf("%f\n", predictedLabels[i]);
		}
		printf("]\n");
	} else {
		printf("predictedLabels is null\n");
	}

	return 0;
}
