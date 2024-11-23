#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdint.h>
#include <cublas_v2.h>
#include <curand.h>
#include <string>
#include <functional>
#include "matMul.h"

#ifdef CUBLAS_API_H_
static const char *_cublasGetError(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "<unknown>";
}
#endif

#ifdef CURAND_H_
static const char *_curandGetError(curandStatus_t error)
{
    switch (error)
    {
        case CURAND_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CURAND_STATUS_VERSION_MISMATCH:
            return "CURAND_STATUS_VERSION_MISMATCH";

        case CURAND_STATUS_NOT_INITIALIZED:
            return "CURAND_STATUS_NOT_INITIALIZED";

        case CURAND_STATUS_ALLOCATION_FAILED:
            return "CURAND_STATUS_ALLOCATION_FAILED";

        case CURAND_STATUS_TYPE_ERROR:
            return "CURAND_STATUS_TYPE_ERROR";

        case CURAND_STATUS_OUT_OF_RANGE:
            return "CURAND_STATUS_OUT_OF_RANGE";

        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

        case CURAND_STATUS_LAUNCH_FAILURE:
            return "CURAND_STATUS_LAUNCH_FAILURE";

        case CURAND_STATUS_PREEXISTING_FAILURE:
            return "CURAND_STATUS_PREEXISTING_FAILURE";

        case CURAND_STATUS_INITIALIZATION_FAILED:
            return "CURAND_STATUS_INITIALIZATION_FAILED";

        case CURAND_STATUS_ARCH_MISMATCH:
            return "CURAND_STATUS_ARCH_MISMATCH";

        case CURAND_STATUS_INTERNAL_ERROR:
            return "CURAND_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}
#endif

#define CUDA_CHECK_ERROR(expr)                                                                                          \
expr;                                                                                                                   \
{                                                                                                                       \
    cudaError_t err = cudaGetLastError();                                                                               \
    if (err != 0)                                                                                                       \
    {                                                                                                                   \
        fprintf(stderr, "%s:%d - %s failed with cuda error: %s\n", __FILE__, __LINE__, #expr, cudaGetErrorString(err)); \
    }                                                                                                                   \
}

#define CUBLAS_CHECK_ERROR(expr)                                                                                            \
{                                                                                                                           \
    cublasStatus_t status = expr;                                                                                           \
    if (status != CUBLAS_STATUS_SUCCESS)                                                                                    \
    {                                                                                                                       \
        fprintf(stderr, "%s:%d - %s failed with cublas error: %s\n", __FILE__, __LINE__, #expr, _cublasGetError(status));   \
    }                                                                                                                       \
}

#define CURAND_CHECK_ERROR(expr)                                                                                            \
{                                                                                                                           \
    curandStatus_t status = expr;                                                                                           \
    if (status != CURAND_STATUS_SUCCESS)                                                                                    \
    {                                                                                                                       \
        fprintf(stderr, "%s:%d - %s failed with curand error: %s\n", __FILE__, __LINE__, #expr, _curandGetError(status));   \
    }                                                                                                                       \
}

#define CHECK_OR_PRINT(expr)                                                            \
{                                                                                       \
    if (!(expr))                                                                        \
    {                                                                                   \
        fprintf(stderr, "%s:%d - %s expression failed\n", __FILE__, __LINE__, #expr);   \
    }                                                                                   \
}

#define EPSILON 0.1f
static bool _validateMatMulResults(float *golden, float *mat, size_t dim)
{
    for (size_t i = 0; i < dim; i++)
    {
        for (size_t j = 0; j < dim; j++)
        {
            size_t idx = i * dim + j;
            if (fabs(golden[idx] - mat[idx]) > EPSILON)
            {
                fprintf(stderr, "row: %lu col: %lu -- expected: %f actual: %f\n", i, j, golden[idx], mat[idx]);
                return false;
            }
        }
    }

    return true;
}

#define DIV_CEIL(num, div) (((num) + (div) - 1) / (div))

struct CudaDeviceProperties
{
    int smVerMajor;
    int smVerMinor;
    int warpSize;
    int maxThreadsPerBlock;
    int maxBlockDimX;
    int maxBlockDimY;
    int maxGridDimX;
    int maxGridDimY;
    int maxSmemPerBlock;
    int totalConstMem;
    int maxRegistersPerBlock;
    int l2CacheSize;
    int smCount;
};
CudaDeviceProperties g_cudaDeviceProperties;

void cudaDeviceGetProperties(int deviceIdx)
{
    CUDA_CHECK_ERROR(cudaDeviceGetAttribute(&g_cudaDeviceProperties.smVerMajor, cudaDevAttrComputeCapabilityMajor, deviceIdx));
    CUDA_CHECK_ERROR(cudaDeviceGetAttribute(&g_cudaDeviceProperties.smVerMinor, cudaDevAttrComputeCapabilityMinor, deviceIdx));
    CUDA_CHECK_ERROR(cudaDeviceGetAttribute(&g_cudaDeviceProperties.warpSize, cudaDevAttrWarpSize, deviceIdx));
    CUDA_CHECK_ERROR(cudaDeviceGetAttribute(&g_cudaDeviceProperties.maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, deviceIdx));
    CUDA_CHECK_ERROR(cudaDeviceGetAttribute(&g_cudaDeviceProperties.maxBlockDimX, cudaDevAttrMaxBlockDimX, deviceIdx));
    CUDA_CHECK_ERROR(cudaDeviceGetAttribute(&g_cudaDeviceProperties.maxBlockDimY, cudaDevAttrMaxBlockDimY, deviceIdx));
    CUDA_CHECK_ERROR(cudaDeviceGetAttribute(&g_cudaDeviceProperties.maxGridDimX, cudaDevAttrMaxGridDimX, deviceIdx));
    CUDA_CHECK_ERROR(cudaDeviceGetAttribute(&g_cudaDeviceProperties.maxGridDimY, cudaDevAttrMaxGridDimY, deviceIdx));
    CUDA_CHECK_ERROR(cudaDeviceGetAttribute(&g_cudaDeviceProperties.maxSmemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, deviceIdx));
    CUDA_CHECK_ERROR(cudaDeviceGetAttribute(&g_cudaDeviceProperties.totalConstMem, cudaDevAttrTotalConstantMemory, deviceIdx));
    CUDA_CHECK_ERROR(cudaDeviceGetAttribute(&g_cudaDeviceProperties.maxRegistersPerBlock, cudaDevAttrMaxRegistersPerBlock, deviceIdx));
    CUDA_CHECK_ERROR(cudaDeviceGetAttribute(&g_cudaDeviceProperties.l2CacheSize, cudaDevAttrL2CacheSize, deviceIdx));
    CUDA_CHECK_ERROR(cudaDeviceGetAttribute(&g_cudaDeviceProperties.smCount, cudaDevAttrMultiProcessorCount, deviceIdx));

    printf("======= DEVICE %d ATTRIBUTES ======\n", deviceIdx);
    printf("smVerMajor:             %d\n", g_cudaDeviceProperties.smVerMajor);
    printf("smVerMinor:             %d\n", g_cudaDeviceProperties.smVerMinor);
    printf("warpSize:               %d\n", g_cudaDeviceProperties.warpSize);
    printf("maxThreadsPerBlock:     %d\n", g_cudaDeviceProperties.maxThreadsPerBlock);
    printf("maxBlockDimX:           %d\n", g_cudaDeviceProperties.maxBlockDimX);
    printf("maxBlockDimY:           %d\n", g_cudaDeviceProperties.maxBlockDimY);
    printf("maxGridDimX:            %d\n", g_cudaDeviceProperties.maxGridDimX);
    printf("maxGridDimY:            %d\n", g_cudaDeviceProperties.maxGridDimY);
    printf("maxSmemPerBlock:        %d\n", g_cudaDeviceProperties.maxSmemPerBlock);
    printf("totalConstMem:          %d\n", g_cudaDeviceProperties.totalConstMem);
    printf("maxRegistersPerBlock:   %d\n", g_cudaDeviceProperties.maxRegistersPerBlock);
    printf("l2CacheSize:            %d\n", g_cudaDeviceProperties.l2CacheSize);
    printf("smCount:                %d\n", g_cudaDeviceProperties.smCount);
    printf("======= DEVICE %d ATTRIBUTES ======\n\n", deviceIdx);
}

enum MatMulImplType
{
    MAT_MUL_IMPL_CUBLAS,
    MAT_MUL_IMPL_NAIVE,
    MAT_MUL_IMPL_COUNT,
};

struct MatMulImpl
{
    const std::string typeStr;
    std::function<void(float *, float *, float *, size_t)> implWrapper;
    float elapsedMs = 0.0f;
};

float runCuBLASMatMul
(
    float *d_matrixA,
    float *d_matrixB,
    float *d_matrixRes,
    size_t matDim
)
{
    float time = 0.0f;
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK_ERROR(cudaEventCreate(&start));
    CUDA_CHECK_ERROR(cudaEventCreate(&stop));

    cublasHandle_t handle;
    CUBLAS_CHECK_ERROR(cublasCreate(&handle));

    static const float scaling = 1.0f;
    CUDA_CHECK_ERROR(cudaEventRecord(start));
    CUBLAS_CHECK_ERROR(
        cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            matDim, matDim, matDim,
            &scaling,
            d_matrixA, matDim,
            d_matrixB, matDim,
            &scaling,
            d_matrixRes, matDim
        )
    );
    CUDA_CHECK_ERROR(cudaEventRecord(stop));
    CUDA_CHECK_ERROR(cudaEventSynchronize(stop));
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&time, start, stop));

    return time;
}

void runNaiveMatMul
(
    float *d_matrixA,
    float *d_matrixB,
    float *d_matrixRes,
    size_t matDim
)
{
    dim3 blockDim = {{
        .x = (uint32_t)g_cudaDeviceProperties.warpSize,
        .y = (uint32_t)(g_cudaDeviceProperties.maxThreadsPerBlock / g_cudaDeviceProperties.warpSize),
        .z = 1,
    }};

    dim3 gridDim = {{
        .x = (uint32_t)DIV_CEIL(matDim , blockDim.x),
        .y = (uint32_t)DIV_CEIL(matDim , blockDim.y),
        .z = 1,
    }};

    cuMatrixMulNaive<<<gridDim, blockDim>>>(d_matrixA, d_matrixB, d_matrixRes, matDim);
}

void runImplementations
(
    float *d_matrixA,
    float *d_matrixB,
    float *d_matrixRes,
    float *h_matrixRes,
    float *h_goldenResult,
    size_t matDim,
    std::array<MatMulImpl, MAT_MUL_IMPL_COUNT> &implArray
)
{
    cudaEvent_t start;
    cudaEvent_t stop;
    float cublasTime = 0.0f;
    CUDA_CHECK_ERROR(cudaEventCreate(&start));
    CUDA_CHECK_ERROR(cudaEventCreate(&stop));

    const size_t matSize = matDim * matDim;

    cublasTime = runCuBLASMatMul(d_matrixA, d_matrixB, d_matrixRes, matDim);
    cudaMemcpy(h_goldenResult, d_matrixRes, matSize * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < MAT_MUL_IMPL_COUNT; i++)
    {
        CUDA_CHECK_ERROR(cudaMemset2D(d_matrixRes, matDim * sizeof(float), 0, matDim * sizeof(float), matDim))
        MatMulImpl &impl = implArray[i];
        CUDA_CHECK_ERROR(cudaEventRecord(start));
        impl.implWrapper(d_matrixA, d_matrixB, d_matrixRes, matDim);
        CUDA_CHECK_ERROR(cudaEventRecord(stop));
        CUDA_CHECK_ERROR(cudaEventSynchronize(stop));
        CUDA_CHECK_ERROR(cudaEventElapsedTime(&impl.elapsedMs, start, stop));

        cudaMemcpy(h_matrixRes, d_matrixRes, matSize * sizeof(float), cudaMemcpyDeviceToHost);

        if (_validateMatMulResults(h_goldenResult, h_matrixRes, matDim))
        {
            printf("Implementation: %s took %.1fms to execute, %.2fX as long as CUBLAS\n",
                impl.typeStr.c_str(),
                impl.elapsedMs,
                impl.elapsedMs / cublasTime);
        }
        else
        {
            printf("Errors detected in implementation: %s\n", impl.typeStr.c_str());
        }
    }
}

int main(int argc, char **argv)
{
    static const size_t N = 16384;
    static const size_t matSize = N * N;

    float *h_pMatrixRes = NULL;
    float *h_pMatrixResGolden = NULL;

    float *d_pMatrixA = NULL;
    float *d_pMatrixB = NULL;
    float *d_pMatrixRes = NULL;
    float *d_pMatrixResGolden = NULL;

    cudaDeviceGetProperties(0);

    CUDA_CHECK_ERROR(cudaMallocHost(&h_pMatrixRes, sizeof(float) * matSize));
    CUDA_CHECK_ERROR(cudaMallocHost(&h_pMatrixResGolden, sizeof(float) * matSize));

    CUDA_CHECK_ERROR(cudaMalloc(&d_pMatrixA, sizeof(float) * matSize));
    CUDA_CHECK_ERROR(cudaMalloc(&d_pMatrixB, sizeof(float) * matSize));
    CUDA_CHECK_ERROR(cudaMalloc(&d_pMatrixRes, sizeof(float) * matSize));
    CUDA_CHECK_ERROR(cudaMalloc(&d_pMatrixResGolden, sizeof(float) * matSize));

    curandGenerator_t generator;
    CURAND_CHECK_ERROR(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_XORWOW));
    // Don't need randomization across runs
    CURAND_CHECK_ERROR(curandSetPseudoRandomGeneratorSeed(generator, 0ULL));

    CURAND_CHECK_ERROR(curandGenerateUniform(generator, d_pMatrixA, matSize));
    CURAND_CHECK_ERROR(curandGenerateUniform(generator, d_pMatrixB, matSize));
    CUDA_CHECK_ERROR(cudaMemset2D(d_pMatrixRes, N * sizeof(float), 0, N * sizeof(float), N))
    CUDA_CHECK_ERROR(cudaMemset2D(d_pMatrixResGolden, N * sizeof(float), 0, N * sizeof(float), N))

    curandStatus_t curandGenerateNormal(
        curandGenerator_t generator, 
        float *outputPtr, size_t n, 
        float mean, float stddev);


    std::array<MatMulImpl, MAT_MUL_IMPL_COUNT> implArray = {{
        {.typeStr = "CUBLAS", .implWrapper = runCuBLASMatMul},
        {.typeStr = "Naive", .implWrapper = runNaiveMatMul},
    }};

    runImplementations(
        d_pMatrixA, 
        d_pMatrixB, 
        d_pMatrixRes, 
        h_pMatrixRes, 
        h_pMatrixResGolden, 
        N, 
        implArray);
}