// csv_data_loader.h
#pragma once
#include <fstream>
#include <iostream>
#include <assert.h>
#include <sstream>

#include "matrix.h"
#include "data_loader.h"


using namespace std;


class CSVDataLoader: public DataLoader {
	public:
		Matrix loadFromFile(string path);
};
