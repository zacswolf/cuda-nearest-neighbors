// nearest_neighbor_gpu.h
#pragma once
#include "utils.h"
#include "matrix.h"

template <typename T, typename G>
G* gpuNormal(Matrix<T> &trainData, Matrix<G> &trainLabels, Matrix<T> &testData);

template <typename T, typename G>
G* gpuJLGaussian(Matrix<T> &trainData, Matrix<G> &trainLabels, Matrix<T> &testData, int newDim);

template <typename T, typename G>
G* gpuJLBernoulli(Matrix<T> &trainData, Matrix<G> &trainLabels, Matrix<T> &testData, int newDim);

template <typename T, typename G>
G* gpuJLFast(Matrix<T> &trainData, Matrix<G> &trainLabels, Matrix<T> &testData, int newDim);

//#include "nearest_neighbor_gpu.cu"