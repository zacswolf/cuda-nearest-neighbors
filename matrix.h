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
		__host__ __device__ Matrix(int numRows, int numCols): data(new T[numRows*numCols]), numRows(numRows), numCols(numCols), device(0) { };
		// Matrix(int numRows, int numCols, int device): data(new T[numRows*numCols]), numRows(numRows), numCols(numCols), device(device) { };
		Matrix(vector<T> data, int numRows, int numCols): data(data.data()), numRows(numRows), numCols(numCols), device(0) { };

		inline int index(int row, int col);

		void fill(std::normal_distribution<float> distribution);
		void fill(std::bernoulli_distribution distribution);
		void fill(std::uniform_int_distribution<> distribution);

		void fill(T val);

		Matrix<T> transpose();

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

		template <typename G>
		static Matrix<decltype(std::declval<T&>() * std::declval<G&>())> matMulDiagGPU(Matrix<T> &left, Matrix<G> &diag);

		template <typename G>
		static Matrix<decltype(std::declval<T&>() * std::declval<G&>())> matMulDiagSeq(Matrix<T> &left, Matrix<G> &diag);

		static Matrix<T> matMulWalshHadamardGPU(Matrix<T> left);

		static Matrix<T> matMulWalshHadamardSeq(Matrix<T> left);
		
		template <typename G>
		static Matrix<decltype(std::declval<T&>() * std::declval<G&>())> matMulWithOneHotGPU(Matrix<T> left, Matrix<G> oneHot);

		template <typename G>
		static Matrix<decltype(std::declval<T&>() * std::declval<G&>())> matMulWithOneHotSeq(Matrix<T> left, Matrix<G> oneHot);

		static float l2RowDistanceSeq(Matrix &left, int leftRow, Matrix &right, int rightRow);

		void print();

		__host__ __device__ ~Matrix() {
			// delete [] data;
		}
};

#include "matrix.cu"
