// matrix.h
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
class Matrix {
	public:
		T * const data;
		const int numRows;
		const int numCols;
		const int device;

		Matrix(T *data, int numRows, int numCols): data(data), numRows(numRows), numCols(numCols), device(0) { };
		Matrix(T *data, int numRows, int numCols, int device): data(data), numRows(numRows), numCols(numCols), device(device) { };
		Matrix(int numRows, int numCols): data(new T[numRows*numCols]), numRows(numRows), numCols(numCols), device(0) { };
		// Matrix(int numRows, int numCols, int device): data(new T[numRows*numCols]), numRows(numRows), numCols(numCols), device(device) { };
		Matrix(vector<T> data, int numRows, int numCols): data(data.data()), numRows(numRows), numCols(numCols), device(0) { };

		int index(int row, int col);

		void fill(std::normal_distribution<float> distribution);
		void fill(std::bernoulli_distribution distribution);
		void fill(T val);

		// Removes and returns column from data
		pair<Matrix<T>, Matrix<T>> popColumn(int columnIndex);

		template <typename G>
		Matrix<G> convert() {
			G *arr = new G[this->numRows * this->numCols];
			std::copy(this->data, this->data + this->numRows * this->numCols, arr);
			return Matrix<G>(arr, this->numRows, this->numCols);
		}

		Matrix<T> toDevice(int device);

		template <typename G>
		static Matrix<decltype(std::declval<T&>() * std::declval<G&>())> matMulSeq(Matrix<T> &left, Matrix<G> &right);

		template <typename G>
		static Matrix<decltype(std::declval<T&>() * std::declval<G&>())> Matrix<T>::matMulGPU(Matrix<T> &left, Matrix<G> &right);

		static float l2RowDistanceSeq(Matrix &left, int leftRow, Matrix &right, int rightRow);

		void print();

		~Matrix() {
			// delete [] data;
		}
};

#include "matrix.cu"
