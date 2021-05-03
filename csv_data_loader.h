// csv_data_loader.h
#pragma once
#include <fstream>
#include <iostream>
#include <sstream>

#include "matrix.h"
#include "data_loader_interface.h"


using namespace std;

template <typename T>
class CSVDataLoader: public DataLoaderInterface<T> {
	public:
		Matrix<T> loadFromFile(string path);
};

#include "csv_data_loader.tpp"
