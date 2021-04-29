// nearest_neighbor_sequential.h
#pragma once
#include "random"
#include "utils.h"
#include "matrix.h"

float* seqNormal(Matrix *data, Matrix *labels, Matrix *predictData);
float* seqJLGaussian(Matrix *data, Matrix *labels, Matrix *predictData, float epsilon);
