#include "csv_data_loader.h"

Matrix CSVDataLoader::loadFromFile(string path) {
	vector<float> data;

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

			// the last column is the label
			// labels.push_back(data.back());
			// data.pop_back();

			// Define dim to be dimTemp if we are on the first row
			dim = (!((bool)dim))*dimTemp + ((bool)dim)*dim;
			
			assert(dimTemp == dim);
		}

		inputFile.close();

		// for (int i = 0; i < data.size(); i++){
		// 	printf("%f ", data[i]);
		// }

		return Matrix(data, numPoints, dim);
	} else {
		cerr << "Unable to open input file: " << path << "\n";
		exit(EXIT_FAILURE);
	}
}
