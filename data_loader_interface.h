// data_loader.h
#pragma once
#include <vector>

#include "matrix.h"

class DataLoaderInterface {
	public:
		// returns data, num rows, num columns
		virtual Matrix* loadFromFile(string path) = 0;
};
