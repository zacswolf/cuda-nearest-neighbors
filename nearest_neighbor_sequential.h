// nearest_neighbor_sequential.h
#pragma once
#include "utils.h"
#include "matrix.h"
#include "math.h"

template <typename T, typename G>
G* seqNormal(Matrix<T> &trainData, Matrix<G> &trainLabels, Matrix<T> &testData);

template <typename T, typename G>
G* seqJLGaussian(Matrix<T> &trainData, Matrix<G> &trainLabels, Matrix<T> &testData, int newDim);

template <typename T, typename G>
G* seqJLBernoulli(Matrix<T> &trainData, Matrix<G> &trainLabels, Matrix<T> &testData, int newDim);

template <typename T, typename G>
G* seqJLFast(Matrix<T> &trainData, Matrix<G> &trainLabels, Matrix<T> &testData, int newDim);

#include "nearest_neighbor_sequential.tpp"
