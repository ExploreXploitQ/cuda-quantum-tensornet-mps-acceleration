# Device-resident acceleration for CUDA-Q `tensornet-mps`

This repository contains a proposed internal acceleration path for CUDA-Q's
existing single-GPU `tensornet-mps` target. The public target name, Python and
C++ APIs, target configuration, and MPS environment variables remain unchanged.
Applications continue to use the documented CUDA-Q interface:

```python
import cudaq

cudaq.set_target("tensornet-mps")
```

The work is based on NVIDIA CUDA-Q commit
`3f42bbf0266aefa6c34c7671ffda48541defa4e6` and is maintained in the
`feature/accelerated-tensornet-mps` branch of the CUDA-Q source tree under
`cuda-quantum/`.

## Scope

The goal is to reduce GPU execution overhead in the native, single-GPU,
double-precision MPS simulator without adding another public backend or changing
application code. The implementation keeps MPS site tensors resident on the
GPU while it executes supported gates, analytic expectation values, and direct
sampling.

## Implementation

The new `AcceleratedMPSEngine` stores each MPS site as a double-complex CUDA
allocation and performs supported operations on a nonblocking CUDA stream.

For a one-qubit gate, a CUDA kernel updates the physical dimension of the
affected site tensor directly. For a two-qubit gate, the engine contracts the
two sites with the gate, reshapes the result into a matrix, factorizes it, and
updates the neighboring MPS tensors. Nonadjacent two-qubit gates use SWAP
routing and restore the original logical ordering afterward. A controlled
single-target gate is converted to the corresponding two-qubit matrix and uses
the same path.

The default MPS configuration uses a covariance decomposition on the GPU. The
engine forms the Gram matrix with cuBLAS, computes its eigendecomposition with
`cusolverDnZheevd`, applies the configured absolute and relative truncation
cutoffs, and limits the result to `CUDAQ_MPS_MAX_BOND`. Forming a Gram matrix
squares the condition number, so this path is restricted to cutoff values for
which it has been tested. More demanding numerical configurations remain on
the native cuTensorNet implementation.

Analytic `observe` contracts each Pauli term against the resident MPS. Direct
`sample` constructs right environments and evaluates conditional samples on
the GPU for all requested shots. This sampling implementation belongs to the
standard CUDA-Q `sample` path. It does not share samples across Hamiltonian
terms or introduce a separate diagonal-Hamiltonian API.

## Native fallback

The accelerated state can be exported as native cuTensorNet MPS tensors. When
CUDA-Q encounters an operation or configuration not handled by the new engine,
the simulator materializes a native `TensorNetState` and continues through the
existing implementation. This transition preserves the public behavior instead
of rejecting an otherwise supported CUDA-Q program.

The device-resident path is currently selected when all of the following are
true:

- The target is the double-precision `tensornet-mps` target.
- The state starts from two or more zero-initialized qubits.
- No MPS gauge option is set.
- The SVD algorithm is the default `GESVDJ` algorithm.
- The larger of the absolute and relative cutoffs is at least `1.5e-8`.
- The requested gate is a one-qubit gate, a two-qubit gate, or a single-control
  single-target gate.

CUDA-Q uses the native path for FP32, alternate SVD algorithms, gauge options,
lower cutoffs, noise, finite-shot `observe`, reset, state import or export,
unsupported controlled operations, and other operations that already require
native tensor-network handling. Noisy direct sampling also remains native.

A program may use the accelerated engine for circuit evolution and later switch
to the native path when it requests an unsupported operation. The transition
is one-way for that simulation state, which must be accounted for in benchmark
design and interpretation.

## Build integration

The normal CUDA-Q contributor build command is unchanged:

```bash
bash scripts/build_cudaq.sh
```

This repository contains source code, as does the upstream CUDA-Q GitHub
repository. A fresh clone must be built before it can be imported or executed.
Build products are not committed to Git. Wheels, installers, and runtime images
would be produced separately by the CUDA-Q release pipeline.

The build requires the cuSOLVER development header `cusolverDn.h` to enable the
new factorization path. The development-container package list now includes
`libcusolver-dev`. The higher-level development Dockerfile also installs the
matching package when an existing CUDA-Q development image contains the
cuSOLVER runtime but not its headers.

CMake detects the header and cuSOLVER library and defines
`CUDAQ_MPS_HAS_CUSOLVER` for the double-precision MPS plugin. If the development
files are unavailable, CUDA-Q still builds and `tensornet-mps` uses the native
implementation. There is no Conda dependency and no application-specific build
step.

The build continues to produce the standard files:

```text
lib/libnvqir-tensornet-mps.so
targets/tensornet-mps.yml
```

## Source changes

| File | Purpose |
| --- | --- |
| `cuda-quantum/runtime/nvqir/cutensornet/accelerated_mps_engine.h` | Internal device-resident MPS interface |
| `cuda-quantum/runtime/nvqir/cutensornet/accelerated_mps_engine.cu` | CUDA kernels, gate updates, factorization, expectation values, sampling, and native tensor export |
| `cuda-quantum/runtime/nvqir/cutensornet/simulator_mps.h` | Selection of the accelerated path and transition to the native simulator |
| `cuda-quantum/runtime/nvqir/cutensornet/CMakeLists.txt` | CUDA source integration and optional cuSOLVER detection |
| `cuda-quantum/python/tests/kernel/test_state_mps.py` | Standard API, fallback, and allocation compatibility tests |
| `cuda-quantum/docker/build/devcontainer.Dockerfile` | cuSOLVER development package in newly built development images |
| `cuda-quantum/docker/build/cudaq.dev.Dockerfile` | Header installation for published development images that lack it |

## Correctness validation

Three tests were added to CUDA-Q's existing MPS Python test file:

- A five-qubit circuit with local and nonlocal controlled gates compares the
  state vector and analytic energy against `qpp-cpu`.
- A circuit containing `exp_pauli` and reset checks the transition from the
  device-resident state to the native implementation.
- Qubit allocation after prior gate execution checks that appended qubits have
  the same state and ordering as `qpp-cpu`.

## Performance status

The current application-level check used a 97-qubit linear-entanglement circuit
with four ansatz repetitions, 970 parameters, a 3,016-term Hamiltonian, maximum
bond dimension 16, and 1,000 finite shots on an NVIDIA L40S. Four consecutive
optimizer evaluations took 14.31, 14.32, 14.28, and 14.28 seconds, for a mean of
14.30 seconds.

## References

- [CUDA-Q documentation](https://nvidia.github.io/cuda-quantum/latest/)
- [CUDA-Q tensor-network simulator documentation](https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/tnsims.html)
- [CUDA-Q contributor build instructions](https://github.com/NVIDIA/cuda-quantum/blob/main/Building.md)
- [CUDA-Q development environment setup](https://github.com/NVIDIA/cuda-quantum/blob/main/Dev_Setup.md)
