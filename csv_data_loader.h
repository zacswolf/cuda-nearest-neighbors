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
	private:
		int labelColumnIndex;
		bool hasLabelColumn = false;

	public:
		CSVDataLoader() : hasLabelColumn(false) { }
		CSVDataLoader(int labelColumnIndex) : labelColumnIndex(labelColumnIndex), hasLabelColumn(true) { }

		Matrix<T> load(string path);
};

#include "csv_data_loader.tpp"
