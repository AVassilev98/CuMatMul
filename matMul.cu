#include "matMul.h"
#include <stdio.h>

__global__ void cuMatrixMulNaive(float *a, float *b, float *result, size_t N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= N || y >= N)
    {
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < N; i++)
    {
        sum += a[i * N + x] * b[y * N + i];
    }
    result[y * N + x] = sum;
}