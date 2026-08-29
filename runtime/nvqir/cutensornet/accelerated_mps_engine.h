/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nvqir {

/// A single-GPU, device-resident MPS execution engine used as an internal fast
/// path by the native tensornet-mps simulator. The public CUDA-Q simulator and
/// target interfaces remain unchanged.
class AcceleratedMPSEngine {
public:
  // Flattened kernel indices contain four times the product of two bond
  // dimensions and must fit in a signed 32-bit integer.
  static constexpr int64_t MaximumBond = 23170;
  // The covariance factorization squares the matrix condition number.
  static constexpr double MinimumStableCutoff = 1.5e-8;

  struct DeviceTensor {
    void *data = nullptr;
    std::vector<int64_t> extents;
  };

  AcceleratedMPSEngine(std::size_t numQubits, int64_t maxBond, double absCutoff,
                       double relCutoff, int svdAlgorithm);
  ~AcceleratedMPSEngine();

  AcceleratedMPSEngine(const AcceleratedMPSEngine &) = delete;
  AcceleratedMPSEngine &operator=(const AcceleratedMPSEngine &) = delete;
  AcceleratedMPSEngine(AcceleratedMPSEngine &&) noexcept;
  AcceleratedMPSEngine &operator=(AcceleratedMPSEngine &&) noexcept;

  std::size_t numQubits() const;
  void appendZeroQubits(std::size_t count);
  void resetZero();
  void synchronize();

  void applyOneQubit(const std::vector<std::complex<double>> &matrix,
                     std::size_t qubit);
  void applyTwoQubit(const std::vector<std::complex<double>> &matrix,
                     std::size_t firstQubit, std::size_t secondQubit);

  std::complex<double> expectation(const std::string &pauliWord);
  bool supportsSampling(std::size_t shots) const;
  std::vector<std::string> sample(std::size_t shots, std::size_t seed);

  /// Return newly allocated tensors in the same Fortran-order layout and
  /// extent convention as cutensornetStateFinalizeMPS. The caller owns them.
  std::vector<DeviceTensor> exportNativeTensors();

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace nvqir
