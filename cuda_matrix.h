// cuda_matrix.h
#pragma once
#include <vector>
#include <iostream>
#include "random"
#include "vector"
#include "utility"

#include "matrix_interface.h"
#include "exceptions.h"
#include <cassert>

using namespace std;

template <typename T>
class CudaMatrix{
	public:
		T * const data;
		const int numRows;
		const int numCols;

		CudaMatrix(T *data, int numRows, int numCols): data(data), numRows(numRows), numCols(numCols) { };
		CudaMatrix(int numRows, int numCols): data(new T[numRows*numCols]), numRows(numRows), numCols(numCols) { };
		CudaMatrix(vector<T> data, int numRows, int numCols): data(data.data()), numRows(numRows), numCols(numCols) { };


		int index(int row, int col);

		void fill(std::normal_distribution<float> distribution);
		void fill(std::bernoulli_distribution distribution);

		// Removes and returns column from data
		pair<CudaMatrix<T>, CudaMatrix<T>> popColumn(int columnIndex);

		template <typename G>
		CudaMatrix<G> convert() {
			G *arr = new G[this->numRows * this->numCols];
			std::copy(this->data, this->data + this->numRows * this->numCols, arr);
			return CudaMatrix<G>(arr, this->numRows, this->numCols);
		}

		template <typename G>
		static CudaMatrix<decltype(std::declval<T&>() * std::declval<G&>())> matMulSeq(CudaMatrix<T> &left, CudaMatrix<G> &right);

		static float l2RowDistanceCUDA(CudaMatrix &left, int leftRow, CudaMatrix &right, int rightRow);

		void print();

		~CudaMatrix() {
			// delete [] data;
		}
};

#include "cuda_matrix.cu"
