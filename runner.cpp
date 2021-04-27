#include <iostream>
#include <unistd.h>
#include <fstream>

#include "utils.h"
#include "matrix.h"
#include "csv_data_loader.h"




int main(int argc, char *argv[]) { 
	// Arguments 
	std::string inputDatasetPath;
	std::string inputPredictDataPath;
    bool gpu = false;
    Mode mode = Mode::NORMAL;
	
	// Load Arguments
	int opt;
	while ((opt = getopt(argc, argv, "d:p:gm:")) != -1) {
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
		default:
			break;
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

    #if DEBUG
        data.print();
    #endif

	// Split labels from dataset
    Matrix labels = data.popColumn(-1); 

    #if DEBUG
        data.print();
        labels.print();
    #endif

	// Get points to classify
	Matrix predictData = dl.loadFromFile(inputPredictDataPath);
	#if DEBUG
        predictData.print();
    #endif

    return 1;
}
