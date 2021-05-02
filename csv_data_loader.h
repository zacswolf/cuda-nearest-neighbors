// csv_data_loader.h
#pragma once
#include <fstream>
#include <iostream>
#include <sstream>

#include "matrix.h"
#include "data_loader_interface.h"


using namespace std;

class CSVDataLoader: public DataLoaderInterface {
	public:
		Matrix loadFromFile(string path);
};
