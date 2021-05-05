#include "csv_data_loader.h"

template<typename T>
Matrix<T> CSVDataLoader<T>::load(string path) {
	// vector<float> *data = new vector<float>;
	vector<float> data;
	// vector<float> c = *dataH;

	// Read CSV file, TODO: abstract this
	// Process Input File
	ifstream inputFile;
	inputFile.open(path);

	string line;
	string word;

	int numPoints = 0;
	int dimTemp;
	int dim = 0;

	if (inputFile.is_open()) {
		while (getline(inputFile,line)) {
			numPoints++;

			stringstream s(line);
			dimTemp = 0;
			while (getline(s, word, ',')) {
				dimTemp++;
				data.push_back(stod(word));
			}

			// Define dim to be dimTemp if we are on the first row
			dim = (!((bool)dim))*dimTemp + ((bool)dim)*dim;
			
			if (dimTemp != dim){
				throw std::invalid_argument(path + " dimentions are not the same on row " + to_string(numPoints));
			}
		}

		inputFile.close();

		float *arr = new float[numPoints*dim];
		std::copy(data.begin(), data.end(), arr);

		// if (this->hasLabelColumn) {

		// 	// Split labels from train dataset
		// 	return Matrix<T>(arr, numPoints, dim).popColumn(-1);
		// } else {
		return Matrix<T>(arr, numPoints, dim);
		// }
	} else {
		cerr << "Unable to open input file: " << path << "\n";
		exit(EXIT_FAILURE);
	}
}

