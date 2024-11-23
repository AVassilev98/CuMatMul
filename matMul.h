#pragma once    
#include <stddef.h>

__global__ void cuMatrixMulNaive(float *a, float *b, float *result, size_t N);