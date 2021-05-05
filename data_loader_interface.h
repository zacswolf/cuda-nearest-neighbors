// data_loader.h
#pragma once
#include "matrix.h"

template <typename T>
class DataLoaderInterface {
	public:
		// returns data, num rows, num columns
		virtual Matrix<T> load(string path) = 0;
};
