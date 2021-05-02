// nearest_neighbor_sequential.h
#pragma once
#include "utils.h"
#include "matrix.h"

float* seqNormal(Matrix &data, Matrix &labels, Matrix &predictData);
float* seqJLGaussian(Matrix &data, Matrix &labels, Matrix &predictData, int newDim);
float* seqJLBernoulli(Matrix &data, Matrix &labels, Matrix &predictData, int newDim);
float* seqJLFast(Matrix &data, Matrix &labels, Matrix &predictData, int newDim);
