/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "accelerated_mps_engine.h"

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cutensornet.h>

#ifdef CUDAQ_MPS_HAS_CUSOLVER
#include <cublas_v2.h>
#include <cusolverDn.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <new>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

using DeviceComplex = cuDoubleComplex;
constexpr int Threads = 256;

[[noreturn]] void throwCuda(cudaError_t status, const char *expression,
                            const char *file, int line) {
  std::ostringstream message;
  message << "CUDA failure in " << expression << " at " << file << ':' << line
          << ": " << cudaGetErrorString(status);
  throw std::runtime_error(message.str());
}

[[noreturn]] void throwCutensornet(cutensornetStatus_t status,
                                   const char *expression, const char *file,
                                   int line) {
  std::ostringstream message;
  message << "cuTensorNet failure in " << expression << " at " << file << ':'
          << line << ": " << cutensornetGetErrorString(status);
  throw std::runtime_error(message.str());
}

#ifdef CUDAQ_MPS_HAS_CUSOLVER
[[noreturn]] void throwCublas(cublasStatus_t status, const char *expression,
                              const char *file, int line) {
  std::ostringstream message;
  message << "cuBLAS failure in " << expression << " at " << file << ':' << line
          << ": " << cublasGetStatusString(status);
  throw std::runtime_error(message.str());
}

[[noreturn]] void throwCusolver(cusolverStatus_t status, const char *expression,
                                const char *file, int line) {
  std::ostringstream message;
  message << "cuSOLVER failure in " << expression << " at " << file << ':'
          << line << ": status " << static_cast<int>(status);
  throw std::runtime_error(message.str());
}
#endif

#define ACCEL_CUDA(expression)                                                 \
  do {                                                                         \
    const auto status = (expression);                                          \
    if (status != cudaSuccess)                                                 \
      throwCuda(status, #expression, __FILE__, __LINE__);                      \
  } while (false)

#define ACCEL_CUTN(expression)                                                 \
  do {                                                                         \
    const auto status = (expression);                                          \
    if (status != CUTENSORNET_STATUS_SUCCESS)                                  \
      throwCutensornet(status, #expression, __FILE__, __LINE__);               \
  } while (false)

#ifdef CUDAQ_MPS_HAS_CUSOLVER
#define ACCEL_CUBLAS(expression)                                               \
  do {                                                                         \
    const auto status = (expression);                                          \
    if (status != CUBLAS_STATUS_SUCCESS)                                       \
      throwCublas(status, #expression, __FILE__, __LINE__);                    \
  } while (false)

#define ACCEL_CUSOLVER(expression)                                             \
  do {                                                                         \
    const auto status = (expression);                                          \
    if (status != CUSOLVER_STATUS_SUCCESS)                                     \
      throwCusolver(status, #expression, __FILE__, __LINE__);                  \
  } while (false)
#endif

#define ACCEL_KERNEL() ACCEL_CUDA(cudaGetLastError())

const DeviceComplex One = make_cuDoubleComplex(1.0, 0.0);
const DeviceComplex Zero = make_cuDoubleComplex(0.0, 0.0);

struct Gate4 {
  DeviceComplex value[16];
};

__global__ void applyOneQubitKernel(DeviceComplex *tensor, DeviceComplex u00,
                                    DeviceComplex u01, DeviceComplex u10,
                                    DeviceComplex u11, int left, int right) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= left * right)
    return;
  const int r = index % right;
  const int l = index / right;
  const auto a0 = tensor[(l * 2) * right + r];
  const auto a1 = tensor[(l * 2 + 1) * right + r];
  tensor[(l * 2) * right + r] = cuCadd(cuCmul(u00, a0), cuCmul(u01, a1));
  tensor[(l * 2 + 1) * right + r] = cuCadd(cuCmul(u10, a0), cuCmul(u11, a1));
}

__global__ void buildTwoSiteMatrixKernel(DeviceComplex *matrix,
                                         const DeviceComplex *leftTensor,
                                         const DeviceComplex *rightTensor,
                                         Gate4 gate, int leftBond,
                                         int centerBond, int rightBond) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = leftBond * 4 * rightBond;
  if (index >= count)
    return;

  int value = index;
  const int r = value % rightBond;
  value /= rightBond;
  const int outputSecond = value % 2;
  value /= 2;
  const int outputFirst = value % 2;
  const int l = value / 2;

  DeviceComplex result = make_cuDoubleComplex(0.0, 0.0);
  for (int inputFirst = 0; inputFirst < 2; ++inputFirst) {
    for (int inputSecond = 0; inputSecond < 2; ++inputSecond) {
      const auto gateValue = gate.value[(outputFirst * 2 + outputSecond) * 4 +
                                        inputFirst * 2 + inputSecond];
      DeviceComplex contraction = make_cuDoubleComplex(0.0, 0.0);
      for (int center = 0; center < centerBond; ++center) {
        contraction = cuCadd(
            contraction,
            cuCmul(leftTensor[(l * 2 + inputFirst) * centerBond + center],
                   rightTensor[(center * 2 + inputSecond) * rightBond + r]));
      }
      result = cuCadd(result, cuCmul(gateValue, contraction));
    }
  }

  const int row = l * 2 + outputFirst;
  const int column = outputSecond * rightBond + r;
  const int rows = 2 * leftBond;
  matrix[row + static_cast<std::size_t>(column) * rows] = result;
}

__global__ void columnMajorToRowMajorKernel(const DeviceComplex *input,
                                            DeviceComplex *output, int rows,
                                            int columns) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * columns)
    return;
  const int row = index / columns;
  const int column = index % columns;
  output[index] = input[row + static_cast<std::size_t>(column) * rows];
}

#ifdef CUDAQ_MPS_HAS_CUSOLVER
__global__ void
selectDescendingEigenvectorsKernel(const DeviceComplex *ascending,
                                   DeviceComplex *selected, int dimension,
                                   int rank) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= dimension * rank)
    return;
  const int row = index % dimension;
  const int column = index / dimension;
  selected[row + static_cast<std::size_t>(column) * dimension] =
      ascending[row +
                static_cast<std::size_t>(dimension - 1 - column) * dimension];
}

__global__ void normalizeColumnsKernel(DeviceComplex *matrix, int rows,
                                       int columns,
                                       const double *ascendingEigenvalues,
                                       int eigenvalueCount) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * columns)
    return;
  const int column = index / rows;
  const double eigenvalue = ascendingEigenvalues[eigenvalueCount - 1 - column];
  const double inverse = eigenvalue > 0.0 ? rsqrt(eigenvalue) : 1.0;
  matrix[index].x *= inverse;
  matrix[index].y *= inverse;
}
#endif

__global__ void transferLeftKernel(const DeviceComplex *environment,
                                   const DeviceComplex *tensor,
                                   DeviceComplex *temporary, int left,
                                   int right) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= left * 2 * right)
    return;
  const int rightPrime = index % right;
  const int value = index / right;
  const int physicalPrime = value % 2;
  const int l = value / 2;
  DeviceComplex result = make_cuDoubleComplex(0.0, 0.0);
  for (int leftPrime = 0; leftPrime < left; ++leftPrime)
    result = cuCadd(
        result,
        cuCmul(environment[l * left + leftPrime],
               tensor[(leftPrime * 2 + physicalPrime) * right + rightPrime]));
  temporary[index] = result;
}

__global__ void applyPauliKernel(const DeviceComplex *input,
                                 DeviceComplex *output, int left, int right,
                                 DeviceComplex p00, DeviceComplex p01,
                                 DeviceComplex p10, DeviceComplex p11) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= left * 2 * right)
    return;
  const int r = index % right;
  const int value = index / right;
  const int physical = value % 2;
  const int l = value / 2;
  const auto in0 = input[(l * 2) * right + r];
  const auto in1 = input[(l * 2 + 1) * right + r];
  const auto first = physical == 0 ? p00 : p10;
  const auto second = physical == 0 ? p01 : p11;
  output[index] = cuCadd(cuCmul(first, in0), cuCmul(second, in1));
}

__global__ void transferRightKernel(const DeviceComplex *tensor,
                                    const DeviceComplex *temporary,
                                    DeviceComplex *environment, int left,
                                    int right) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= right * right)
    return;
  const int rightPrime = index % right;
  const int r = index / right;
  DeviceComplex result = make_cuDoubleComplex(0.0, 0.0);
  for (int row = 0; row < left * 2; ++row)
    result = cuCadd(result, cuCmul(cuConj(tensor[row * right + r]),
                                   temporary[row * right + rightPrime]));
  environment[index] = result;
}

__global__ void rightEnvironmentStageKernel(const DeviceComplex *tensor,
                                            const DeviceComplex *rightNext,
                                            DeviceComplex *temporary, int left,
                                            int right) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= left * 2 * right)
    return;
  const int rightPrime = index % right;
  const int value = index / right;
  const int physical = value % 2;
  const int l = value / 2;
  DeviceComplex result = make_cuDoubleComplex(0.0, 0.0);
  for (int r = 0; r < right; ++r)
    result = cuCadd(result, cuCmul(tensor[(l * 2 + physical) * right + r],
                                   rightNext[r * right + rightPrime]));
  temporary[index] = result;
}

__global__ void rightEnvironmentFinishKernel(const DeviceComplex *tensor,
                                             const DeviceComplex *temporary,
                                             DeviceComplex *rightEnvironment,
                                             int left, int right) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= left * left)
    return;
  const int leftPrime = index % left;
  const int l = index / left;
  DeviceComplex result = make_cuDoubleComplex(0.0, 0.0);
  for (int physical = 0; physical < 2; ++physical)
    for (int rightPrime = 0; rightPrime < right; ++rightPrime)
      result = cuCadd(
          result,
          cuCmul(
              temporary[(l * 2 + physical) * right + rightPrime],
              cuConj(tensor[(leftPrime * 2 + physical) * right + rightPrime])));
  rightEnvironment[index] = result;
}

__global__ void initializeShotVectorsKernel(DeviceComplex *vectors,
                                            std::size_t shots) {
  const std::size_t shot =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (shot < shots)
    vectors[shot] = make_cuDoubleComplex(1.0, 0.0);
}

__global__ void sampleCandidatesKernel(const DeviceComplex *vectors,
                                       const DeviceComplex *tensor,
                                       DeviceComplex *candidates,
                                       std::size_t shots, int left, int right) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = shots * 2 * right;
  if (index >= count)
    return;
  const int r = index % right;
  const std::size_t value = index / right;
  const int physical = value % 2;
  const std::size_t shot = value / 2;
  DeviceComplex result = make_cuDoubleComplex(0.0, 0.0);
  for (int l = 0; l < left; ++l)
    result = cuCadd(result, cuCmul(vectors[shot * left + l],
                                   tensor[(l * 2 + physical) * right + r]));
  candidates[index] = result;
}

__global__ void sampleProbabilitiesKernel(const DeviceComplex *candidates,
                                          const DeviceComplex *rightEnvironment,
                                          double *probabilities,
                                          std::size_t shots, int right) {
  const std::size_t pair = blockIdx.x;
  if (pair >= shots * 2)
    return;
  extern __shared__ double partial[];
  double value = 0.0;
  for (int r = threadIdx.x; r < right; r += blockDim.x) {
    const auto *candidate = candidates + pair * right;
    DeviceComplex contracted = make_cuDoubleComplex(0.0, 0.0);
    for (int rightPrime = 0; rightPrime < right; ++rightPrime)
      contracted =
          cuCadd(contracted, cuCmul(rightEnvironment[r * right + rightPrime],
                                    cuConj(candidate[rightPrime])));
    value += cuCmul(candidate[r], contracted).x;
  }
  partial[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    probabilities[pair] = partial[0] > 0.0 ? partial[0] : 0.0;
}

__device__ unsigned long long sampleHash(unsigned long long value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

__global__ void chooseSamplesKernel(const DeviceComplex *candidates,
                                    const double *probabilities,
                                    DeviceComplex *nextVectors, char *bits,
                                    std::size_t shots, int right, int site,
                                    int numSites, std::size_t seed) {
  const std::size_t shot = blockIdx.x;
  if (shot >= shots)
    return;
  __shared__ int selected;
  __shared__ double inverseNorm;
  if (threadIdx.x == 0) {
    double p0 = probabilities[shot * 2];
    double p1 = probabilities[shot * 2 + 1];
    double total = p0 + p1;
    if (total <= 0.0) {
      p0 = p1 = 0.5;
      total = 1.0;
    }
    const auto raw =
        sampleHash((static_cast<unsigned long long>(seed) << 32) ^
                   (static_cast<unsigned long long>(site) << 20) ^ shot);
    const double uniform = static_cast<double>(raw >> 11) * 0x1.0p-53;
    selected = uniform < p0 / total ? 0 : 1;
    const double probability = selected ? p1 : p0;
    inverseNorm = probability > 0.0 ? 1.0 / sqrt(probability) : 1.0;
    bits[shot * numSites + site] = selected ? '1' : '0';
  }
  __syncthreads();
  for (int r = threadIdx.x; r < right; r += blockDim.x) {
    const auto value = candidates[(shot * 2 + selected) * right + r];
    nextVectors[shot * right + r] =
        make_cuDoubleComplex(value.x * inverseNorm, value.y * inverseNorm);
  }
}

__global__ void exportTensorKernel(const DeviceComplex *source,
                                   DeviceComplex *destination, int left,
                                   int right, int nativeSite, int numSites) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= left * 2 * right)
    return;
  const int r = index % right;
  const int value = index / right;
  const int physical = value % 2;
  const int l = value / 2;
  std::size_t destinationIndex = 0;
  if (numSites == 1)
    destinationIndex = physical;
  else if (nativeSite == 0)
    destinationIndex = physical + static_cast<std::size_t>(2) * r;
  else if (nativeSite == numSites - 1)
    destinationIndex = l + static_cast<std::size_t>(left) * physical;
  else
    destinationIndex = l + static_cast<std::size_t>(left) *
                               (physical + static_cast<std::size_t>(2) * r);
  destination[destinationIndex] = source[index];
}

std::vector<std::complex<double>> swapMatrix() {
  std::vector<std::complex<double>> matrix(16, 0.0);
  for (int outputFirst = 0; outputFirst < 2; ++outputFirst)
    for (int outputSecond = 0; outputSecond < 2; ++outputSecond)
      for (int inputFirst = 0; inputFirst < 2; ++inputFirst)
        for (int inputSecond = 0; inputSecond < 2; ++inputSecond)
          if (outputFirst == inputSecond && outputSecond == inputFirst)
            matrix[(outputFirst * 2 + outputSecond) * 4 + inputFirst * 2 +
                   inputSecond] = 1.0;
  return matrix;
}

std::vector<std::complex<double>>
swapOperands(const std::vector<std::complex<double>> &matrix) {
  std::vector<std::complex<double>> swapped(16);
  for (int outputFirst = 0; outputFirst < 2; ++outputFirst)
    for (int outputSecond = 0; outputSecond < 2; ++outputSecond)
      for (int inputFirst = 0; inputFirst < 2; ++inputFirst)
        for (int inputSecond = 0; inputSecond < 2; ++inputSecond)
          swapped[(outputFirst * 2 + outputSecond) * 4 + inputFirst * 2 +
                  inputSecond] = matrix[(outputSecond * 2 + outputFirst) * 4 +
                                        inputSecond * 2 + inputFirst];
  return swapped;
}

void pauliMatrix(char pauli, DeviceComplex &p00, DeviceComplex &p01,
                 DeviceComplex &p10, DeviceComplex &p11) {
  p00 = p01 = p10 = p11 = Zero;
  switch (pauli) {
  case 'X':
    p01 = p10 = One;
    break;
  case 'Y':
    p01 = make_cuDoubleComplex(0.0, -1.0);
    p10 = make_cuDoubleComplex(0.0, 1.0);
    break;
  case 'Z':
    p00 = One;
    p11 = make_cuDoubleComplex(-1.0, 0.0);
    break;
  default:
    p00 = p11 = One;
    break;
  }
}

} // namespace

namespace nvqir {

class AcceleratedMPSEngine::Impl {
public:
  struct Site {
    DeviceComplex *data = nullptr;
    int left = 1;
    int right = 1;
  };

  Impl(std::size_t numQubits, int64_t maxBond, double absCutoff,
       double relCutoff, int svdAlgorithm, std::size_t seed)
      : maxBond(static_cast<int>(maxBond)), absCutoff(absCutoff),
        relCutoff(relCutoff), svdAlgorithm(svdAlgorithm), randomSeed(seed) {
    if (numQubits == 0)
      throw std::invalid_argument(
          "Accelerated MPS requires at least one qubit");
    if (maxBond < 1)
      throw std::invalid_argument("MPS maximum bond must be positive");
    ACCEL_CUDA(cudaGetDevice(&device));
    ACCEL_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    try {
#ifdef CUDAQ_MPS_HAS_CUSOLVER
      if (usesCovarianceSvd()) {
        ACCEL_CUBLAS(cublasCreate(&cublasHandle));
        ACCEL_CUBLAS(cublasSetStream(cublasHandle, stream));
        ACCEL_CUSOLVER(cusolverDnCreate(&cusolverHandle));
        ACCEL_CUSOLVER(cusolverDnSetStream(cusolverHandle, stream));
      }
#endif
      if (!usesCovarianceSvd())
        ACCEL_CUTN(cutensornetCreate(&cutnHandle));
      appendZeroQubits(numQubits);
    } catch (...) {
      release();
      throw;
    }
  }

  ~Impl() { release(); }

  void release() noexcept {
    cudaSetDevice(device);
    if (stream)
      cudaStreamSynchronize(stream);
    for (auto &site : sites)
      if (site.data)
        cudaFree(site.data);
    sites.clear();
    if (cutnHandle)
      cutensornetDestroy(cutnHandle);
#ifdef CUDAQ_MPS_HAS_CUSOLVER
    if (cusolverHandle)
      cusolverDnDestroy(cusolverHandle);
    if (cublasHandle)
      cublasDestroy(cublasHandle);
#endif
    if (stream)
      cudaStreamDestroy(stream);
    cutnHandle = nullptr;
#ifdef CUDAQ_MPS_HAS_CUSOLVER
    cusolverHandle = nullptr;
    cublasHandle = nullptr;
#endif
    stream = nullptr;
  }

  void appendZeroQubits(std::size_t count) {
    ACCEL_CUDA(cudaSetDevice(device));
    const DeviceComplex zeroState[2]{One, Zero};
    for (std::size_t index = 0; index < count; ++index) {
      DeviceComplex *data = nullptr;
      ACCEL_CUDA(cudaMalloc(&data, sizeof(zeroState)));
      ACCEL_CUDA(cudaMemcpyAsync(data, zeroState, sizeof(zeroState),
                                 cudaMemcpyHostToDevice, stream));
      // CUDA-Q assigns appended qubits increasing indices, while the native
      // cuTensorNet MPS tensor list is ordered from the highest qubit to q0.
      sites.insert(sites.begin(), {data, 1, 1});
    }
  }

  void resetZero() {
    ACCEL_CUDA(cudaSetDevice(device));
    ACCEL_CUDA(cudaStreamSynchronize(stream));
    const DeviceComplex zeroState[2]{One, Zero};
    for (auto &site : sites) {
      if (site.left != 1 || site.right != 1) {
        ACCEL_CUDA(cudaFree(site.data));
        ACCEL_CUDA(cudaMalloc(&site.data, sizeof(zeroState)));
      }
      ACCEL_CUDA(cudaMemcpyAsync(site.data, zeroState, sizeof(zeroState),
                                 cudaMemcpyHostToDevice, stream));
      site.left = site.right = 1;
    }
  }

  void applyOneQubit(const std::vector<std::complex<double>> &matrix,
                     std::size_t qubit) {
    if (matrix.size() != 4 || qubit >= sites.size())
      throw std::invalid_argument("Invalid one-qubit MPS gate");
    const std::size_t siteIndex = sites.size() - 1 - qubit;
    auto &site = sites[siteIndex];
    DeviceComplex gate[4];
    for (int index = 0; index < 4; ++index)
      gate[index] =
          make_cuDoubleComplex(matrix[index].real(), matrix[index].imag());
    const int count = site.left * site.right;
    applyOneQubitKernel<<<(count + Threads - 1) / Threads, Threads, 0,
                          stream>>>(site.data, gate[0], gate[1], gate[2],
                                    gate[3], site.left, site.right);
    ACCEL_KERNEL();
  }

  struct SvdFactors {
    DeviceComplex *left = nullptr;
    DeviceComplex *right = nullptr;
    int rank = 0;
  };

  bool usesCovarianceSvd() const {
#ifdef CUDAQ_MPS_HAS_CUSOLVER
    // Forming a Gram matrix squares the condition number. Preserve the native
    // cuTensorNet path when users request cutoffs below that stable range.
    constexpr double minimumStableCutoff = 1.5e-8;
    return svdAlgorithm == CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ &&
           std::max(absCutoff, relCutoff) >= minimumStableCutoff;
#else
    return false;
#endif
  }

#ifdef CUDAQ_MPS_HAS_CUSOLVER
  SvdFactors factorizeCovariance(DeviceComplex *matrix, int rows, int columns) {
    const bool useLeftGram = rows <= columns;
    const int dimension = useLeftGram ? rows : columns;
    DeviceComplex *covariance = nullptr;
    double *eigenvalues = nullptr;
    DeviceComplex *workspace = nullptr;
    int *information = nullptr;
    DeviceComplex *basis = nullptr;
    DeviceComplex *left = nullptr;
    DeviceComplex *right = nullptr;
    try {
      ACCEL_CUDA(cudaMalloc(
          &covariance, sizeof(DeviceComplex) *
                           static_cast<std::size_t>(dimension) * dimension));
      if (useLeftGram) {
        ACCEL_CUBLAS(cublasZgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_C, rows,
                                 rows, columns, &One, matrix, rows, matrix,
                                 rows, &Zero, covariance, rows));
      } else {
        ACCEL_CUBLAS(cublasZgemm(cublasHandle, CUBLAS_OP_C, CUBLAS_OP_N,
                                 columns, columns, rows, &One, matrix, rows,
                                 matrix, rows, &Zero, covariance, columns));
      }

      ACCEL_CUDA(cudaMalloc(&eigenvalues, sizeof(double) * dimension));
      int workspaceElements = 0;
      ACCEL_CUSOLVER(cusolverDnZheevd_bufferSize(
          cusolverHandle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
          dimension, covariance, dimension, eigenvalues, &workspaceElements));
      ACCEL_CUDA(
          cudaMalloc(&workspace, sizeof(DeviceComplex) * workspaceElements));
      ACCEL_CUDA(cudaMalloc(&information, sizeof(int)));
      ACCEL_CUSOLVER(cusolverDnZheevd(
          cusolverHandle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
          dimension, covariance, dimension, eigenvalues, workspace,
          workspaceElements, information));

      int hostInformation = 0;
      std::vector<double> hostEigenvalues(dimension);
      ACCEL_CUDA(cudaMemcpyAsync(&hostInformation, information, sizeof(int),
                                 cudaMemcpyDeviceToHost, stream));
      ACCEL_CUDA(cudaMemcpyAsync(hostEigenvalues.data(), eigenvalues,
                                 sizeof(double) * dimension,
                                 cudaMemcpyDeviceToHost, stream));
      ACCEL_CUDA(cudaStreamSynchronize(stream));
      if (hostInformation != 0)
        throw std::runtime_error("cuSOLVER eigendecomposition failed with "
                                 "information value " +
                                 std::to_string(hostInformation));

      const double maximum = std::max(0.0, hostEigenvalues.back());
      const double threshold =
          std::max(absCutoff * absCutoff, relCutoff * relCutoff * maximum);
      int rank = 0;
      for (auto iterator = hostEigenvalues.rbegin();
           iterator != hostEigenvalues.rend() && *iterator > threshold;
           ++iterator)
        ++rank;
      rank = std::clamp(rank, 1, std::min(maxBond, dimension));

      ACCEL_CUDA(cudaMalloc(&basis, sizeof(DeviceComplex) *
                                        static_cast<std::size_t>(dimension) *
                                        rank));
      selectDescendingEigenvectorsKernel<<<
          (dimension * rank + Threads - 1) / Threads, Threads, 0, stream>>>(
          covariance, basis, dimension, rank);
      ACCEL_KERNEL();

      if (useLeftGram) {
        left = basis;
        basis = nullptr;
      } else {
        ACCEL_CUDA(cudaMalloc(&left, sizeof(DeviceComplex) *
                                         static_cast<std::size_t>(rows) *
                                         rank));
        ACCEL_CUBLAS(cublasZgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, rows,
                                 rank, columns, &One, matrix, rows, basis,
                                 columns, &Zero, left, rows));
        normalizeColumnsKernel<<<(rows * rank + Threads - 1) / Threads, Threads,
                                 0, stream>>>(left, rows, rank, eigenvalues,
                                              dimension);
        ACCEL_KERNEL();
      }

      ACCEL_CUDA(cudaMalloc(&right, sizeof(DeviceComplex) *
                                        static_cast<std::size_t>(rank) *
                                        columns));
      ACCEL_CUBLAS(cublasZgemm(cublasHandle, CUBLAS_OP_C, CUBLAS_OP_N, rank,
                               columns, rows, &One, left, rows, matrix, rows,
                               &Zero, right, rank));
      ACCEL_CUDA(cudaStreamSynchronize(stream));

      cudaFree(basis);
      cudaFree(covariance);
      cudaFree(eigenvalues);
      cudaFree(workspace);
      cudaFree(information);
      return {left, right, rank};
    } catch (...) {
      cudaFree(basis);
      cudaFree(left);
      cudaFree(right);
      cudaFree(covariance);
      cudaFree(eigenvalues);
      cudaFree(workspace);
      cudaFree(information);
      throw;
    }
  }
#endif

  SvdFactors factorizeCutensornet(DeviceComplex *matrix, int rows,
                                  int columns) {
    const int maximumRank = std::min({rows, columns, maxBond});
    const int64_t inputExtents[2]{rows, columns};
    int64_t leftExtents[2]{rows, maximumRank};
    int64_t rightExtents[2]{maximumRank, columns};
    const int32_t inputModes[2]{0, 1};
    const int32_t leftModes[2]{0, 2};
    const int32_t rightModes[2]{2, 1};

    cutensornetTensorDescriptor_t inputDescriptor = nullptr;
    cutensornetTensorDescriptor_t leftDescriptor = nullptr;
    cutensornetTensorDescriptor_t rightDescriptor = nullptr;
    cutensornetTensorSVDConfig_t configuration = nullptr;
    cutensornetTensorSVDInfo_t information = nullptr;
    cutensornetWorkspaceDescriptor_t workspaceDescriptor = nullptr;
    void *deviceWorkspace = nullptr;
    void *hostWorkspace = nullptr;
    DeviceComplex *left = nullptr;
    DeviceComplex *right = nullptr;
    try {
      ACCEL_CUTN(cutensornetCreateTensorDescriptor(
          cutnHandle, 2, inputExtents, nullptr, inputModes, CUDA_C_64F,
          &inputDescriptor));
      ACCEL_CUTN(cutensornetCreateTensorDescriptor(
          cutnHandle, 2, leftExtents, nullptr, leftModes, CUDA_C_64F,
          &leftDescriptor));
      ACCEL_CUTN(cutensornetCreateTensorDescriptor(
          cutnHandle, 2, rightExtents, nullptr, rightModes, CUDA_C_64F,
          &rightDescriptor));
      ACCEL_CUTN(cutensornetCreateTensorSVDConfig(cutnHandle, &configuration));
      ACCEL_CUTN(cutensornetCreateTensorSVDInfo(cutnHandle, &information));

      const auto algorithm =
          static_cast<cutensornetTensorSVDAlgo_t>(svdAlgorithm);
      const auto partition = CUTENSORNET_TENSOR_SVD_PARTITION_SV;
      ACCEL_CUTN(cutensornetTensorSVDConfigSetAttribute(
          cutnHandle, configuration, CUTENSORNET_TENSOR_SVD_CONFIG_ALGO,
          &algorithm, sizeof(algorithm)));
      ACCEL_CUTN(cutensornetTensorSVDConfigSetAttribute(
          cutnHandle, configuration, CUTENSORNET_TENSOR_SVD_CONFIG_ABS_CUTOFF,
          &absCutoff, sizeof(absCutoff)));
      ACCEL_CUTN(cutensornetTensorSVDConfigSetAttribute(
          cutnHandle, configuration, CUTENSORNET_TENSOR_SVD_CONFIG_REL_CUTOFF,
          &relCutoff, sizeof(relCutoff)));
      ACCEL_CUTN(cutensornetTensorSVDConfigSetAttribute(
          cutnHandle, configuration, CUTENSORNET_TENSOR_SVD_CONFIG_S_PARTITION,
          &partition, sizeof(partition)));

      ACCEL_CUTN(cutensornetCreateWorkspaceDescriptor(cutnHandle,
                                                      &workspaceDescriptor));
      ACCEL_CUTN(cutensornetWorkspaceComputeSVDSizes(
          cutnHandle, inputDescriptor, leftDescriptor, rightDescriptor,
          configuration, workspaceDescriptor));

      int64_t deviceWorkspaceSize = 0;
      ACCEL_CUTN(cutensornetWorkspaceGetMemorySize(
          cutnHandle, workspaceDescriptor,
          CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, CUTENSORNET_MEMSPACE_DEVICE,
          CUTENSORNET_WORKSPACE_SCRATCH, &deviceWorkspaceSize));
      if (deviceWorkspaceSize > 0) {
        ACCEL_CUDA(cudaMalloc(&deviceWorkspace, deviceWorkspaceSize));
        ACCEL_CUTN(cutensornetWorkspaceSetMemory(
            cutnHandle, workspaceDescriptor, CUTENSORNET_MEMSPACE_DEVICE,
            CUTENSORNET_WORKSPACE_SCRATCH, deviceWorkspace,
            deviceWorkspaceSize));
      }

      int64_t hostWorkspaceSize = 0;
      ACCEL_CUTN(cutensornetWorkspaceGetMemorySize(
          cutnHandle, workspaceDescriptor,
          CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, CUTENSORNET_MEMSPACE_HOST,
          CUTENSORNET_WORKSPACE_SCRATCH, &hostWorkspaceSize));
      if (hostWorkspaceSize > 0) {
        hostWorkspace = std::malloc(hostWorkspaceSize);
        if (!hostWorkspace)
          throw std::bad_alloc();
        ACCEL_CUTN(cutensornetWorkspaceSetMemory(
            cutnHandle, workspaceDescriptor, CUTENSORNET_MEMSPACE_HOST,
            CUTENSORNET_WORKSPACE_SCRATCH, hostWorkspace, hostWorkspaceSize));
      }

      ACCEL_CUDA(cudaMalloc(&left, sizeof(DeviceComplex) *
                                       static_cast<std::size_t>(rows) *
                                       maximumRank));
      ACCEL_CUDA(cudaMalloc(&right, sizeof(DeviceComplex) *
                                        static_cast<std::size_t>(maximumRank) *
                                        columns));
      ACCEL_CUTN(cutensornetTensorSVD(
          cutnHandle, inputDescriptor, matrix, leftDescriptor, left, nullptr,
          rightDescriptor, right, configuration, information,
          workspaceDescriptor, stream));
      ACCEL_CUDA(cudaStreamSynchronize(stream));

      int64_t reducedRank = maximumRank;
      ACCEL_CUTN(cutensornetTensorSVDInfoGetAttribute(
          cutnHandle, information, CUTENSORNET_TENSOR_SVD_INFO_REDUCED_EXTENT,
          &reducedRank, sizeof(reducedRank)));

      cudaFree(deviceWorkspace);
      std::free(hostWorkspace);
      cutensornetDestroyWorkspaceDescriptor(workspaceDescriptor);
      cutensornetDestroyTensorSVDInfo(information);
      cutensornetDestroyTensorSVDConfig(configuration);
      cutensornetDestroyTensorDescriptor(inputDescriptor);
      cutensornetDestroyTensorDescriptor(leftDescriptor);
      cutensornetDestroyTensorDescriptor(rightDescriptor);
      const int rank = static_cast<int>(reducedRank);
      return {left, right, rank};
    } catch (...) {
      cudaFree(left);
      cudaFree(right);
      cudaFree(deviceWorkspace);
      std::free(hostWorkspace);
      if (workspaceDescriptor)
        cutensornetDestroyWorkspaceDescriptor(workspaceDescriptor);
      if (information)
        cutensornetDestroyTensorSVDInfo(information);
      if (configuration)
        cutensornetDestroyTensorSVDConfig(configuration);
      if (inputDescriptor)
        cutensornetDestroyTensorDescriptor(inputDescriptor);
      if (leftDescriptor)
        cutensornetDestroyTensorDescriptor(leftDescriptor);
      if (rightDescriptor)
        cutensornetDestroyTensorDescriptor(rightDescriptor);
      throw;
    }
  }

  SvdFactors factorize(DeviceComplex *matrix, int rows, int columns) {
#ifdef CUDAQ_MPS_HAS_CUSOLVER
    if (usesCovarianceSvd())
      return factorizeCovariance(matrix, rows, columns);
#endif
    return factorizeCutensornet(matrix, rows, columns);
  }

  void applyAdjacent(const std::vector<std::complex<double>> &matrix,
                     int siteIndex) {
    auto &leftSite = sites[siteIndex];
    auto &rightSite = sites[siteIndex + 1];
    const int leftBond = leftSite.left;
    const int centerBond = leftSite.right;
    const int rightBond = rightSite.right;
    if (centerBond != rightSite.left)
      throw std::runtime_error("Inconsistent accelerated MPS bond");

    Gate4 gate;
    for (int index = 0; index < 16; ++index)
      gate.value[index] =
          make_cuDoubleComplex(matrix[index].real(), matrix[index].imag());
    const int rows = 2 * leftBond;
    const int columns = 2 * rightBond;
    DeviceComplex *twoSiteMatrix = nullptr;
    ACCEL_CUDA(cudaMalloc(&twoSiteMatrix, sizeof(DeviceComplex) *
                                              static_cast<std::size_t>(rows) *
                                              columns));
    try {
      const int count = leftBond * 4 * rightBond;
      buildTwoSiteMatrixKernel<<<(count + Threads - 1) / Threads, Threads, 0,
                                 stream>>>(twoSiteMatrix, leftSite.data,
                                           rightSite.data, gate, leftBond,
                                           centerBond, rightBond);
      ACCEL_KERNEL();
      auto factors = factorize(twoSiteMatrix, rows, columns);

      DeviceComplex *newLeft = nullptr;
      DeviceComplex *newRight = nullptr;
      try {
        ACCEL_CUDA(cudaMalloc(&newLeft, sizeof(DeviceComplex) *
                                            static_cast<std::size_t>(rows) *
                                            factors.rank));
        ACCEL_CUDA(cudaMalloc(
            &newRight, sizeof(DeviceComplex) *
                           static_cast<std::size_t>(factors.rank) * columns));
        columnMajorToRowMajorKernel<<<(rows * factors.rank + Threads - 1) /
                                          Threads,
                                      Threads, 0, stream>>>(
            factors.left, newLeft, rows, factors.rank);
        columnMajorToRowMajorKernel<<<(factors.rank * columns + Threads - 1) /
                                          Threads,
                                      Threads, 0, stream>>>(
            factors.right, newRight, factors.rank, columns);
        ACCEL_KERNEL();
        ACCEL_CUDA(cudaStreamSynchronize(stream));
      } catch (...) {
        cudaFree(newLeft);
        cudaFree(newRight);
        cudaFree(factors.left);
        cudaFree(factors.right);
        throw;
      }

      cudaFree(leftSite.data);
      cudaFree(rightSite.data);
      leftSite = {newLeft, leftBond, factors.rank};
      rightSite = {newRight, factors.rank, rightBond};
      cudaFree(factors.left);
      cudaFree(factors.right);
      cudaFree(twoSiteMatrix);
    } catch (...) {
      cudaFree(twoSiteMatrix);
      throw;
    }
  }

  void applyTwoQubit(const std::vector<std::complex<double>> &matrix,
                     std::size_t firstQubit, std::size_t secondQubit) {
    if (matrix.size() != 16 || firstQubit >= sites.size() ||
        secondQubit >= sites.size() || firstQubit == secondQubit)
      throw std::invalid_argument("Invalid two-qubit MPS gate");
    int first = static_cast<int>(sites.size() - 1 - firstQubit);
    int second = static_cast<int>(sites.size() - 1 - secondQubit);
    if (first > second) {
      applyTwoQubit(swapOperands(matrix), secondQubit, firstQubit);
      return;
    }

    static const auto swap = swapMatrix();
    for (int site = second; site > first + 1; --site)
      applyAdjacent(swap, site - 1);
    applyAdjacent(matrix, first);
    for (int site = first + 1; site < second; ++site)
      applyAdjacent(swap, site);
  }

  std::complex<double> expectation(const std::string &pauliWord) {
    if (pauliWord.size() != sites.size())
      throw std::invalid_argument("Pauli word size does not match MPS state");
    DeviceComplex *environment = nullptr;
    ACCEL_CUDA(cudaMalloc(&environment, sizeof(DeviceComplex)));
    ACCEL_CUDA(cudaMemcpyAsync(environment, &One, sizeof(DeviceComplex),
                               cudaMemcpyHostToDevice, stream));
    try {
      for (std::size_t siteIndex = 0; siteIndex < sites.size(); ++siteIndex) {
        const auto &site = sites[siteIndex];
        DeviceComplex *leftTemporary = nullptr;
        DeviceComplex *pauliTemporary = nullptr;
        DeviceComplex *nextEnvironment = nullptr;
        try {
          const std::size_t tensorElements =
              static_cast<std::size_t>(site.left) * 2 * site.right;
          ACCEL_CUDA(cudaMalloc(&leftTemporary,
                                sizeof(DeviceComplex) * tensorElements));
          ACCEL_CUDA(cudaMalloc(&pauliTemporary,
                                sizeof(DeviceComplex) * tensorElements));
          ACCEL_CUDA(cudaMalloc(&nextEnvironment,
                                sizeof(DeviceComplex) *
                                    static_cast<std::size_t>(site.right) *
                                    site.right));
          DeviceComplex p00, p01, p10, p11;
          pauliMatrix(pauliWord[sites.size() - 1 - siteIndex], p00, p01, p10,
                      p11);
          transferLeftKernel<<<(tensorElements + Threads - 1) / Threads,
                               Threads, 0, stream>>>(
              environment, site.data, leftTemporary, site.left, site.right);
          applyPauliKernel<<<(tensorElements + Threads - 1) / Threads, Threads,
                             0, stream>>>(leftTemporary, pauliTemporary,
                                          site.left, site.right, p00, p01, p10,
                                          p11);
          transferRightKernel<<<(site.right * site.right + Threads - 1) /
                                    Threads,
                                Threads, 0, stream>>>(site.data, pauliTemporary,
                                                      nextEnvironment,
                                                      site.left, site.right);
          ACCEL_KERNEL();
          ACCEL_CUDA(cudaStreamSynchronize(stream));
        } catch (...) {
          cudaFree(leftTemporary);
          cudaFree(pauliTemporary);
          cudaFree(nextEnvironment);
          throw;
        }
        cudaFree(leftTemporary);
        cudaFree(pauliTemporary);
        cudaFree(environment);
        environment = nextEnvironment;
      }
      DeviceComplex result;
      ACCEL_CUDA(cudaMemcpyAsync(&result, environment, sizeof(DeviceComplex),
                                 cudaMemcpyDeviceToHost, stream));
      ACCEL_CUDA(cudaStreamSynchronize(stream));
      cudaFree(environment);
      return {result.x, result.y};
    } catch (...) {
      cudaFree(environment);
      throw;
    }
  }

  std::vector<std::string> sample(std::size_t shots, std::size_t seed) {
    if (shots == 0)
      return {};
    std::vector<DeviceComplex *> rightEnvironments(sites.size() + 1, nullptr);
    DeviceComplex *temporary = nullptr;
    DeviceComplex *vectorsA = nullptr;
    DeviceComplex *vectorsB = nullptr;
    DeviceComplex *candidates = nullptr;
    double *probabilities = nullptr;
    char *deviceBits = nullptr;
    try {
      ACCEL_CUDA(cudaMalloc(&rightEnvironments.back(), sizeof(DeviceComplex)));
      ACCEL_CUDA(cudaMemcpyAsync(rightEnvironments.back(), &One,
                                 sizeof(DeviceComplex), cudaMemcpyHostToDevice,
                                 stream));
      std::size_t maximumTensorElements = 1;
      int maximumBond = 1;
      for (const auto &site : sites) {
        maximumTensorElements =
            std::max(maximumTensorElements,
                     static_cast<std::size_t>(site.left) * 2 * site.right);
        maximumBond = std::max({maximumBond, site.left, site.right});
      }
      ACCEL_CUDA(cudaMalloc(&temporary,
                            sizeof(DeviceComplex) * maximumTensorElements));
      for (int siteIndex = static_cast<int>(sites.size()) - 1; siteIndex >= 0;
           --siteIndex) {
        const auto &site = sites[siteIndex];
        ACCEL_CUDA(cudaMalloc(&rightEnvironments[siteIndex],
                              sizeof(DeviceComplex) *
                                  static_cast<std::size_t>(site.left) *
                                  site.left));
        const int elements = site.left * 2 * site.right;
        rightEnvironmentStageKernel<<<(elements + Threads - 1) / Threads,
                                      Threads, 0, stream>>>(
            site.data, rightEnvironments[siteIndex + 1], temporary, site.left,
            site.right);
        rightEnvironmentFinishKernel<<<(site.left * site.left + Threads - 1) /
                                           Threads,
                                       Threads, 0, stream>>>(
            site.data, temporary, rightEnvironments[siteIndex], site.left,
            site.right);
        ACCEL_KERNEL();
      }

      const std::size_t vectorElements = shots * maximumBond;
      ACCEL_CUDA(cudaMalloc(&vectorsA, sizeof(DeviceComplex) * vectorElements));
      ACCEL_CUDA(cudaMalloc(&vectorsB, sizeof(DeviceComplex) * vectorElements));
      ACCEL_CUDA(cudaMalloc(&candidates,
                            sizeof(DeviceComplex) * shots * 2 * maximumBond));
      ACCEL_CUDA(cudaMalloc(&probabilities, sizeof(double) * shots * 2));
      ACCEL_CUDA(cudaMalloc(&deviceBits, shots * sites.size()));
      initializeShotVectorsKernel<<<(shots + Threads - 1) / Threads, Threads, 0,
                                    stream>>>(vectorsA, shots);
      ACCEL_KERNEL();

      auto *current = vectorsA;
      auto *next = vectorsB;
      for (std::size_t siteIndex = 0; siteIndex < sites.size(); ++siteIndex) {
        const auto &site = sites[siteIndex];
        const std::size_t candidateCount = shots * 2 * site.right;
        sampleCandidatesKernel<<<(candidateCount + Threads - 1) / Threads,
                                 Threads, 0, stream>>>(
            current, site.data, candidates, shots, site.left, site.right);
        int probabilityThreads = 32;
        while (probabilityThreads < site.right && probabilityThreads < Threads)
          probabilityThreads <<= 1;
        sampleProbabilitiesKernel<<<shots * 2, probabilityThreads,
                                    sizeof(double) * probabilityThreads,
                                    stream>>>(candidates,
                                              rightEnvironments[siteIndex + 1],
                                              probabilities, shots, site.right);
        chooseSamplesKernel<<<shots, Threads, 0, stream>>>(
            candidates, probabilities, next, deviceBits, shots, site.right,
            static_cast<int>(siteIndex), static_cast<int>(sites.size()), seed);
        ACCEL_KERNEL();
        std::swap(current, next);
      }

      std::vector<char> hostBits(shots * sites.size());
      ACCEL_CUDA(cudaMemcpyAsync(hostBits.data(), deviceBits, hostBits.size(),
                                 cudaMemcpyDeviceToHost, stream));
      ACCEL_CUDA(cudaStreamSynchronize(stream));
      std::vector<std::string> output(shots, std::string(sites.size(), '0'));
      for (std::size_t shot = 0; shot < shots; ++shot)
        for (std::size_t qubit = 0; qubit < sites.size(); ++qubit)
          output[shot][qubit] =
              hostBits[shot * sites.size() + (sites.size() - 1 - qubit)];

      for (auto *pointer : rightEnvironments)
        cudaFree(pointer);
      cudaFree(temporary);
      cudaFree(vectorsA);
      cudaFree(vectorsB);
      cudaFree(candidates);
      cudaFree(probabilities);
      cudaFree(deviceBits);
      return output;
    } catch (...) {
      for (auto *pointer : rightEnvironments)
        cudaFree(pointer);
      cudaFree(temporary);
      cudaFree(vectorsA);
      cudaFree(vectorsB);
      cudaFree(candidates);
      cudaFree(probabilities);
      cudaFree(deviceBits);
      throw;
    }
  }

  std::vector<AcceleratedMPSEngine::DeviceTensor> exportNativeTensors() {
    std::vector<AcceleratedMPSEngine::DeviceTensor> output;
    output.reserve(sites.size());
    try {
      for (std::size_t siteIndex = 0; siteIndex < sites.size(); ++siteIndex) {
        const auto &site = sites[siteIndex];
        const std::size_t elements =
            static_cast<std::size_t>(site.left) * 2 * site.right;
        std::vector<int64_t> extents;
        if (sites.size() == 1)
          extents = {2};
        else if (siteIndex == 0)
          extents = {2, site.right};
        else if (siteIndex == sites.size() - 1)
          extents = {site.left, 2};
        else
          extents = {site.left, 2, site.right};
        output.push_back({nullptr, std::move(extents)});
        ACCEL_CUDA(
            cudaMalloc(&output.back().data, sizeof(DeviceComplex) * elements));
        exportTensorKernel<<<(elements + Threads - 1) / Threads, Threads, 0,
                             stream>>>(
            site.data, static_cast<DeviceComplex *>(output.back().data),
            site.left, site.right, static_cast<int>(siteIndex),
            static_cast<int>(sites.size()));
        ACCEL_KERNEL();
      }
      ACCEL_CUDA(cudaStreamSynchronize(stream));
      return output;
    } catch (...) {
      for (auto &tensor : output)
        cudaFree(tensor.data);
      throw;
    }
  }

  int device = 0;
  int maxBond = 64;
  double absCutoff = 1e-5;
  double relCutoff = 1e-5;
  int svdAlgorithm = static_cast<int>(CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ);
  std::size_t randomSeed = 0;
  cudaStream_t stream = nullptr;
#ifdef CUDAQ_MPS_HAS_CUSOLVER
  cublasHandle_t cublasHandle = nullptr;
  cusolverDnHandle_t cusolverHandle = nullptr;
#endif
  cutensornetHandle_t cutnHandle = nullptr;
  std::vector<Site> sites;
};

AcceleratedMPSEngine::AcceleratedMPSEngine(std::size_t numQubits,
                                           int64_t maxBond, double absCutoff,
                                           double relCutoff, int svdAlgorithm,
                                           std::size_t randomSeed)
    : impl(std::make_unique<Impl>(numQubits, maxBond, absCutoff, relCutoff,
                                  svdAlgorithm, randomSeed)) {}

AcceleratedMPSEngine::~AcceleratedMPSEngine() = default;
AcceleratedMPSEngine::AcceleratedMPSEngine(AcceleratedMPSEngine &&) noexcept =
    default;
AcceleratedMPSEngine &
AcceleratedMPSEngine::operator=(AcceleratedMPSEngine &&) noexcept = default;

std::size_t AcceleratedMPSEngine::numQubits() const {
  return impl->sites.size();
}

void AcceleratedMPSEngine::appendZeroQubits(std::size_t count) {
  impl->appendZeroQubits(count);
}

void AcceleratedMPSEngine::resetZero() { impl->resetZero(); }

void AcceleratedMPSEngine::synchronize() {
  ACCEL_CUDA(cudaSetDevice(impl->device));
  ACCEL_CUDA(cudaStreamSynchronize(impl->stream));
}

void AcceleratedMPSEngine::setRandomSeed(std::size_t seed) {
  impl->randomSeed = seed;
}

void AcceleratedMPSEngine::applyOneQubit(
    const std::vector<std::complex<double>> &matrix, std::size_t qubit) {
  impl->applyOneQubit(matrix, qubit);
}

void AcceleratedMPSEngine::applyTwoQubit(
    const std::vector<std::complex<double>> &matrix, std::size_t firstQubit,
    std::size_t secondQubit) {
  impl->applyTwoQubit(matrix, firstQubit, secondQubit);
}

std::complex<double>
AcceleratedMPSEngine::expectation(const std::string &pauliWord) {
  return impl->expectation(pauliWord);
}

std::vector<std::string> AcceleratedMPSEngine::sample(std::size_t shots,
                                                      std::size_t seed) {
  return impl->sample(shots, seed);
}

std::vector<AcceleratedMPSEngine::DeviceTensor>
AcceleratedMPSEngine::exportNativeTensors() {
  return impl->exportNativeTensors();
}

} // namespace nvqir
