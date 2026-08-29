/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "accelerated_mps_engine.h"
#include "mps_simulation_state.h"
#include "simulator_cutensornet.h"
#include <charconv>
#include <errno.h>
#include <limits>
#include <type_traits>

namespace nvqir {
template <typename ScalarType = double>
class SimulatorMPS : public SimulatorTensorNetBase<ScalarType> {
  MPSSettings m_settings;
  std::vector<MPSTensor> m_mpsTensors_d;
  std::unique_ptr<AcceleratedMPSEngine> m_acceleratedMps;
  std::size_t m_pendingZeroQubits = 0;

#ifdef CUDAQ_MPS_HAS_CUSOLVER
  static constexpr bool accelerationAvailable =
      std::is_same_v<ScalarType, double>;
#else
  static constexpr bool accelerationAvailable = false;
#endif

  bool accelerationConfigurationSupported() const {
    return !m_settings.gaugeOption.has_value() &&
           m_settings.svdAlgo == CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ &&
           m_settings.maxBond <= AcceleratedMPSEngine::MaximumBond &&
           std::max(m_settings.absCutoff, m_settings.relCutoff) >=
               AcceleratedMPSEngine::MinimumStableCutoff;
  }

  void initializePendingState(bool allowAcceleration) {
    if (m_pendingZeroQubits == 0)
      return;

    if constexpr (accelerationAvailable) {
      if (allowAcceleration && m_pendingZeroQubits > 1 &&
          accelerationConfigurationSupported()) {
        m_acceleratedMps = std::make_unique<AcceleratedMPSEngine>(
            m_pendingZeroQubits, m_settings.maxBond, m_settings.absCutoff,
            m_settings.relCutoff, static_cast<int>(m_settings.svdAlgo));
        CUDAQ_INFO("[SimulatorMPS] enabled device-resident MPS execution for "
                   "{} qubits.",
                   m_pendingZeroQubits);
        m_pendingZeroQubits = 0;
        return;
      }
    }

    m_state = std::make_unique<TensorNetState<ScalarType>>(
        m_pendingZeroQubits, scratchPad, m_cutnHandle, m_randomEngine);
    m_pendingZeroQubits = 0;
  }

  void materializeAcceleratedState() {
    initializePendingState(false);
    if (!m_acceleratedMps)
      return;

    auto exported = m_acceleratedMps->exportNativeTensors();
    try {
      std::vector<MPSTensor> tensors;
      tensors.reserve(exported.size());
      for (auto &tensor : exported)
        tensors.emplace_back(tensor.data, std::move(tensor.extents));

      m_state = TensorNetState<ScalarType>::createFromMpsTensors(
          tensors, scratchPad, m_cutnHandle, m_randomEngine);
      m_state->m_tempDevicePtrs.reserve(tensors.size());
      for (std::size_t i = 0; i < tensors.size(); ++i) {
        auto &tensor = tensors[i];
        m_state->m_tempDevicePtrs.emplace_back(
            tensor.deviceData,
            typename TensorNetState<ScalarType>::TempDevicePtrDeleter{});
        // Ownership has moved to m_state. A later exception must only free
        // tensors that have not yet been transferred.
        exported[i].data = nullptr;
      }
    } catch (...) {
      m_state.reset();
      for (auto &tensor : exported)
        cudaFree(tensor.data);
      throw;
    }
    m_acceleratedMps.reset();
  }

  bool canAccelerate(const typename nvqir::CircuitSimulatorBase<
                     ScalarType>::GateApplicationTask &task) const {
    if (!m_acceleratedMps)
      return false;
    if (task.controls.empty())
      return (task.targets.size() == 1 && task.matrix.size() == 4) ||
             (task.targets.size() == 2 && task.matrix.size() == 16);
    return task.controls.size() == 1 && task.targets.size() == 1 &&
           task.matrix.size() == 4;
  }

public:
  using GateApplicationTask =
      typename nvqir::CircuitSimulatorBase<ScalarType>::GateApplicationTask;
  using SimulatorTensorNetBase<ScalarType>::m_cutnHandle;
  using SimulatorTensorNetBase<
      ScalarType>::m_maxControlledRankForFullTensorExpansion;
  using SimulatorTensorNetBase<ScalarType>::m_state;
  using SimulatorTensorNetBase<ScalarType>::scratchPad;
  using SimulatorTensorNetBase<ScalarType>::m_randomEngine;
  using SimulatorTensorNetBase<ScalarType>::m_reuseContractionPathObserve;
  SimulatorMPS() : SimulatorTensorNetBase<ScalarType>() {}

  void applyGate(const GateApplicationTask &task) override {
    initializePendingState(true);
    if (canAccelerate(task)) {
      if (task.controls.empty() && task.targets.size() == 1) {
        std::vector<std::complex<double>> matrix(task.matrix.begin(),
                                                 task.matrix.end());
        m_acceleratedMps->applyOneQubit(matrix, task.targets.front());
        return;
      }

      if (task.controls.empty()) {
        std::vector<std::complex<double>> matrix(task.matrix.begin(),
                                                 task.matrix.end());
        m_acceleratedMps->applyTwoQubit(matrix, task.targets[0],
                                        task.targets[1]);
        return;
      }

      std::vector<std::complex<double>> targetMatrix(task.matrix.begin(),
                                                     task.matrix.end());
      auto controlledMatrix = generateFullGateTensor(1, targetMatrix);
      m_acceleratedMps->applyTwoQubit(controlledMatrix, task.controls.front(),
                                      task.targets.front());
      return;
    }

    materializeAcceleratedState();
    SimulatorTensorNetBase<ScalarType>::applyGate(task);
  }

  void applyNoiseChannel(const std::string_view gateName,
                         const std::vector<std::size_t> &controls,
                         const std::vector<std::size_t> &targets,
                         const std::vector<double> &params) override {
    materializeAcceleratedState();
    SimulatorTensorNetBase<ScalarType>::applyNoiseChannel(gateName, controls,
                                                          targets, params);
  }

  void applyNoise(const cudaq::kraus_channel &channel,
                  const std::vector<std::size_t> &targets) override {
    materializeAcceleratedState();
    SimulatorTensorNetBase<ScalarType>::applyNoise(channel, targets);
  }

  bool measureQubit(const std::size_t qubitIdx) override {
    materializeAcceleratedState();
    return SimulatorTensorNetBase<ScalarType>::measureQubit(qubitIdx);
  }

  void resetQubit(const std::size_t qubitIdx) override {
    materializeAcceleratedState();
    SimulatorTensorNetBase<ScalarType>::resetQubit(qubitIdx);
  }

  void synchronize() override {
    if (m_acceleratedMps)
      m_acceleratedMps->synchronize();
    else
      SimulatorTensorNetBase<ScalarType>::synchronize();
  }

  virtual void prepareQubitTensorState() override {
    LOG_API_TIME();
    materializeAcceleratedState();
    // Clean up previously factorized MPS tensors
    for (auto &tensor : m_mpsTensors_d) {
      HANDLE_CUDA_ERROR(cudaFree(tensor.deviceData));
    }
    m_mpsTensors_d.clear();
    // Factorize the state:
    if (m_state->getNumQubits() > 1)
      m_mpsTensors_d = m_state->factorizeMPS(
          m_settings.maxBond, m_settings.absCutoff, m_settings.relCutoff,
          m_settings.svdAlgo, m_settings.gaugeOption);
  }

  virtual std::size_t calculateStateDim(const std::size_t numQubits) override {
    return numQubits;
  }

  virtual void
  addQubitsToState(const cudaq::SimulationState &in_state) override {
    LOG_API_TIME();
    materializeAcceleratedState();
    const MPSSimulationState<ScalarType> *const casted =
        dynamic_cast<const MPSSimulationState<ScalarType> *>(&in_state);
    if (!casted)
      throw std::invalid_argument(
          "[SimulatorMPS simulator] Incompatible state input");
    if (!m_state) {
      std::vector<MPSTensor> copiedTensors;
      copiedTensors.reserve(casted->getMpsTensors().size());
      for (const auto &mpsTensor : casted->getMpsTensors()) {
        std::vector<int64_t> extents = mpsTensor.extents;
        const auto numElements =
            std::reduce(extents.begin(), extents.end(), 1, std::multiplies());
        const auto tensorSizeBytes =
            sizeof(std::complex<ScalarType>) * numElements;
        void *mpsTensorCopy{nullptr};
        HANDLE_CUDA_ERROR(cudaMalloc(&mpsTensorCopy, tensorSizeBytes));
        HANDLE_CUDA_ERROR(cudaMemcpy(mpsTensorCopy, mpsTensor.deviceData,
                                     tensorSizeBytes, cudaMemcpyDefault));
        copiedTensors.emplace_back(MPSTensor(mpsTensorCopy, extents));
      }

      m_state = TensorNetState<ScalarType>::createFromMpsTensors(
          copiedTensors, scratchPad, m_cutnHandle, m_randomEngine);
      for (const auto &mpsTensor : copiedTensors) {
        m_state->m_tempDevicePtrs.emplace_back(
            mpsTensor.deviceData,
            typename TensorNetState<ScalarType>::TempDevicePtrDeleter{});
      }
    } else {
      // Expand an existing state: Append MPS tensors
      // Factor the existing state
      auto tensors = m_state->factorizeMPS(
          m_settings.maxBond, m_settings.absCutoff, m_settings.relCutoff,
          m_settings.svdAlgo, m_settings.gaugeOption);
      // The right most MPS tensor needs to have one more extra leg (no longer
      // the boundary tensor).
      tensors.back().extents.emplace_back(1);
      auto mpsTensors = casted->getMpsTensors();
      for (std::size_t i = 0; i < mpsTensors.size(); ++i) {
        auto &tensor = mpsTensors[i];
        std::vector<int64_t> extents = tensor.extents;
        if (i == 0) {
          // First tensor: add a bond (dim 1 since no entanglement) to the
          // existing state
          extents.insert(extents.begin(), 1);
        }
        const auto numElements =
            std::reduce(extents.begin(), extents.end(), 1, std::multiplies());
        const auto tensorSizeBytes =
            sizeof(std::complex<ScalarType>) * numElements;
        void *mpsTensor{nullptr};
        HANDLE_CUDA_ERROR(cudaMalloc(&mpsTensor, tensorSizeBytes));
        HANDLE_CUDA_ERROR(cudaMemcpy(mpsTensor, tensor.deviceData,
                                     tensorSizeBytes, cudaMemcpyDefault));
        tensors.emplace_back(MPSTensor(mpsTensor, extents));
      }
      m_state = TensorNetState<ScalarType>::createFromMpsTensors(
          tensors, scratchPad, m_cutnHandle, m_randomEngine);
    }
  }

  template <typename T>
  std::vector<std::complex<T>> generateXX(double theta) {
    const T halfTheta = theta / 2.;
    const std::complex<T> cos = std::cos(halfTheta);
    const std::complex<T> isin = {0., std::sin(halfTheta)};
    // Row-major
    return {cos, 0.,    0.,  -isin, 0.,    cos, -isin, 0.,
            0.,  -isin, cos, 0.,    -isin, 0.,  0.,    cos};
  };

  template <typename T>
  std::vector<std::complex<T>> generateYY(double theta) {
    const T halfTheta = theta / 2.;
    const std::complex<T> cos = std::cos(halfTheta);
    const std::complex<T> isin = {0., std::sin(halfTheta)};
    // Row-major
    return {cos, 0.,    0.,  isin, 0.,   cos, -isin, 0.,
            0.,  -isin, cos, 0.,   isin, 0.,  0.,    cos};
  };

  template <typename T>
  std::vector<std::complex<T>> generateZZ(double theta) {
    const std::complex<T> itheta2 = {0., static_cast<T>(theta / 2.0)};
    const std::complex<T> exp_itheta2 = std::exp(itheta2);
    const std::complex<T> exp_minus_itheta2 =
        std::exp(static_cast<T>(-1.0) * itheta2);
    // Row-major
    return {exp_minus_itheta2, 0., 0., 0., 0., exp_itheta2,      0., 0., 0., 0.,
            exp_itheta2,       0., 0., 0., 0., exp_minus_itheta2};
  };

  virtual void applyExpPauli(double theta,
                             const std::vector<std::size_t> &controls,
                             const std::vector<std::size_t> &qubitIds,
                             const cudaq::spin_op_term &op) override {
    if (cudaq::isInTracerMode()) {
      nvqir::CircuitSimulator::applyExpPauli(theta, controls, qubitIds, op);
      return;
    }
    // Special handling for equivalence of Rxx(theta), Ryy(theta), Rzz(theta)
    // expressed as exp_pauli.
    //  Note: for MPS, the runtime is ~ linear with the number of 2-body gates
    //  (gate split procedure).
    // Hence, we check if this is a Rxx(theta), Ryy(theta), or Rzz(theta), which
    // are commonly-used gates and apply the operation directly (the base
    // decomposition will result in 2 CNOT gates).
    const auto shouldHandlePauliOp = [](const std::string &pauli_word) -> bool {
      return pauli_word == "XX" || pauli_word == "YY" || pauli_word == "ZZ";
    };

    // FIXME: the implementation here assumes that  the spin op term is not
    // a general spin op term, but really just a pauli word; it's coefficient is
    // silently ignored. This works because it was actually constructed from a
    // pauli word - we should just pass that one along.
    auto pauli_word = op.get_pauli_word();
    if (controls.empty() && qubitIds.size() == 2 &&
        shouldHandlePauliOp(pauli_word)) {
      this->flushGateQueue();
      CUDAQ_INFO("[SimulatorMPS] (apply) exp(i*{}*{}) ({}, {}).", theta,
                 op.to_string(), qubitIds[0], qubitIds[1]);
      const GateApplicationTask task = [&]() {
        // Note: Rxx(angle) ==  exp(-i*angle/2 XX)
        // i.e., exp(i*theta XX) == Rxx(-2 * theta)
        if (pauli_word == "XX") {
          // Note: use a special name so that the gate matrix caching procedure
          // works properly.
          return GateApplicationTask(
              "Rxx", generateXX<ScalarType>(-2.0 * theta), {}, qubitIds,
              {static_cast<ScalarType>(theta)});
        } else if (pauli_word == "YY") {
          return GateApplicationTask(
              "Ryy", generateYY<ScalarType>(-2.0 * theta), {}, qubitIds,
              {static_cast<ScalarType>(theta)});
        } else if (pauli_word == "ZZ") {
          return GateApplicationTask(
              "Rzz", generateZZ<ScalarType>(-2.0 * theta), {}, qubitIds,
              {static_cast<ScalarType>(theta)});
        }
        __builtin_unreachable();
      }();
      this->applyGate(task);
      return;
    }
    // Let the base class to handle this Pauli rotation
    materializeAcceleratedState();
    SimulatorTensorNetBase<ScalarType>::applyExpPauli(theta, controls, qubitIds,
                                                      op);
  }

  // Helper to compute expectation value from a bit string distribution
  static double computeExpValFromDistribution(
      const std::unordered_map<std::string, std::size_t> &distribution,
      int shots) {
    double expVal = 0.0;
    // Compute the expectation value from the distribution
    for (auto &kv : distribution) {
      auto par = cudaq::sample_result::has_even_parity(kv.first);
      auto p = kv.second / (double)shots;
      if (!par) {
        p = -p;
      }
      expVal += p;
    }
    return expVal;
  };

  // Set up the MPS factorization before trajectory simulation run loop.
  // We only need to do cutensornetStateFinalizeMPS once
  void setUpFactorizeForTrajectoryRuns() {
    for (auto &tensor : m_mpsTensors_d) {
      HANDLE_CUDA_ERROR(cudaFree(tensor.deviceData));
    }

    if (m_state->hasGeneralChannelApplied() && m_state->getNumQubits() <= 1)
      throw std::runtime_error(
          "MPS noisy simulation currently does not support the case where "
          "number of qubit is equal to 1");
    m_mpsTensors_d.clear();
    m_mpsTensors_d = m_state->setupMPSFactorize(
        m_settings.maxBond, m_settings.absCutoff, m_settings.relCutoff,
        m_settings.svdAlgo, m_settings.gaugeOption);
  }

  /// @brief Sample a subset of qubits
  cudaq::ExecutionResult sample(const std::vector<std::size_t> &measuredBits,
                                const int shots,
                                bool includeSequentialData = true) override {
    initializePendingState(true);
    auto executionContext = cudaq::getExecutionContext();

    const bool hasNoise = this->getNoiseModel() != nullptr;
    if (m_acceleratedMps && !hasNoise &&
        (shots < 1 || m_acceleratedMps->supportsSampling(shots))) {
      LOG_API_TIME();
      if (shots < 1) {
        const std::string allZ(m_acceleratedMps->numQubits(), 'Z');
        return cudaq::ExecutionResult(
            {}, m_acceleratedMps->expectation(allZ).real());
      }

      auto bitStrings = m_acceleratedMps->sample(
          static_cast<std::size_t>(shots), m_randomEngine());
      cudaq::ExecutionResult counts;
      for (const auto &fullBitString : bitStrings) {
        std::string measured;
        measured.reserve(measuredBits.size());
        for (const auto qubit : measuredBits)
          measured.push_back(fullBitString.at(qubit));
        if (includeSequentialData)
          counts.appendResult(measured, 1);
        else
          counts.counts[measured]++;
      }

      double expectationValue = 0.0;
      for (const auto &[bitString, count] : counts.counts) {
        const double probability =
            static_cast<double>(count) / static_cast<double>(shots);
        expectationValue += cudaq::sample_result::has_even_parity(bitString)
                                ? probability
                                : -probability;
      }
      counts.expectationValue = expectationValue;
      return counts;
    }

    // Noise and any other non-accelerated sampling path require a native
    // TensorNetState.
    materializeAcceleratedState();
    if (!hasNoise || shots < 1)
      return SimulatorTensorNetBase<ScalarType>::sample(measuredBits, shots,
                                                        includeSequentialData);

    LOG_API_TIME();
    cudaq::ExecutionResult counts;
    std::vector<int32_t> measuredBitIds(measuredBits.begin(),
                                        measuredBits.end());

    setUpFactorizeForTrajectoryRuns();
    std::map<std::vector<int64_t>, std::pair<cutensornetStateSampler_t,
                                             cutensornetWorkspaceDescriptor_t>>
        samplerCache;
    for (int i = 0; i < shots; ++i) {
      // As the Kraus operator sampling may change the MPS state, we need to
      // re-compute the factorization in each trajectory.
      m_state->computeMPSFactorize(m_mpsTensors_d);
      std::vector<int64_t> samplerKey;
      for (const auto &tensor : m_mpsTensors_d)
        samplerKey.insert(samplerKey.end(), tensor.extents.begin(),
                          tensor.extents.end());

      auto iter = samplerCache.find(samplerKey);
      if (iter == samplerCache.end()) {
        const auto [itInsert, success] = samplerCache.insert(
            {samplerKey, m_state->prepareSample(measuredBitIds)});
        assert(success);
        iter = itInsert;
      }

      assert(iter != samplerCache.end());
      auto &[sampler, workDesc] = iter->second;
      const auto samples = m_state->executeSample(
          sampler, workDesc, measuredBitIds, 1, requireCacheWorkspace());
      assert(samples.size() == 1);
      for (const auto &[bitString, count] : samples) {
        if (includeSequentialData)
          counts.appendResult(bitString, count);
        else
          counts.counts[bitString] += count;
      }
    }

    for (const auto &[k, v] : samplerCache) {
      auto &[sampler, workDesc] = v;
      // Destroy the workspace descriptor
      HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
      // Destroy the quantum circuit sampler
      HANDLE_CUTN_ERROR(cutensornetDestroySampler(sampler));
    }

    counts.expectationValue =
        computeExpValFromDistribution(counts.counts, shots);

    return counts;
  }

  // Helper to prepare term-by-term data from a spin op
  static std::tuple<std::vector<std::string>, std::vector<cudaq::spin_op_term>>
  prepareSpinOpTermData(const cudaq::spin_op &ham) {
    std::vector<std::string> termStrs;
    std::vector<cudaq::spin_op_term> prods;
    termStrs.reserve(ham.num_terms());
    prods.reserve(ham.num_terms());
    for (auto &&term : ham) {
      termStrs.emplace_back(term.get_term_id());
      prods.push_back(std::move(term));
    }
    return std::make_tuple(termStrs, prods);
  }

  cudaq::observe_result observe(const cudaq::spin_op &ham) override {
    initializePendingState(true);
    assert(cudaq::spin_op::canonicalize(ham) == ham);
    auto executionContext = cudaq::getExecutionContext();

    LOG_API_TIME();
    const bool hasNoise = this->getNoiseModel() != nullptr;
    const bool finiteShots =
        executionContext && executionContext->shots > 0 &&
        executionContext->shots != std::numeric_limits<std::size_t>::max();
    if (m_acceleratedMps && !hasNoise && !finiteShots) {
      if (ham.num_terms() == 0) {
        std::vector<cudaq::ExecutionResult> emptyResults;
        cudaq::sample_result emptyData(0.0, emptyResults);
        return cudaq::observe_result(0.0, ham, emptyData);
      }

      std::complex<double> total = 0.0;
      std::vector<cudaq::ExecutionResult> termResults;
      termResults.reserve(ham.num_terms());
      for (const auto &term : ham) {
        const auto pauliWord =
            term.get_pauli_word(m_acceleratedMps->numQubits());
        const auto termValue = term.evaluate_coefficient() *
                               m_acceleratedMps->expectation(pauliWord);
        total += termValue;
        termResults.emplace_back(
            cudaq::ExecutionResult({}, term.get_term_id(), termValue.real()));
      }

      if (!m_reuseContractionPathObserve) {
        return cudaq::observe_result(
            total.real(), ham,
            cudaq::sample_result(
                cudaq::ExecutionResult({}, ham.to_string(), total.real())));
      }

      cudaq::sample_result perTermData(total.real(), termResults);
      return cudaq::observe_result(total.real(), ham, perTermData);
    }

    materializeAcceleratedState();
    // If no noise, just use base class implementation.
    if (!hasNoise)
      return SimulatorTensorNetBase<ScalarType>::observe(ham);

    setUpFactorizeForTrajectoryRuns();
    const std::size_t numObserveTrajectories =
        executionContext->numberTrajectories.has_value()
            ? executionContext->numberTrajectories.value()
            : TensorNetState<ScalarType>::g_numberTrajectoriesForObserve;

    auto [termStrs, terms] = prepareSpinOpTermData(ham);
    std::vector<std::complex<double>> termExpVals(terms.size(), 0.0);

    for (std::size_t i = 0; i < numObserveTrajectories; ++i) {
      // As the Kraus operator sampling may change the MPS state, we need to
      // re-compute the factorization in each trajectory.
      m_state->computeMPSFactorize(m_mpsTensors_d);
      // We run a single trajectory for MPS as the final MPS form depends on the
      // randomly-selected noise op.
      const auto trajTermExpVals = m_state->computeExpVals(terms, 1);

      for (std::size_t idx = 0; idx < terms.size(); ++idx) {
        termExpVals[idx] += (trajTermExpVals[idx] /
                             static_cast<ScalarType>(numObserveTrajectories));
      }
    }
    std::complex<double> expVal = 0.0;
    // Construct per-term data in the final observe_result
    std::vector<cudaq::ExecutionResult> results;
    results.reserve(terms.size());

    for (std::size_t i = 0; i < terms.size(); ++i) {
      expVal += termExpVals[i];
      results.emplace_back(
          cudaq::ExecutionResult({}, termStrs[i], termExpVals[i].real()));
    }

    cudaq::sample_result perTermData(expVal.real(), results);
    return cudaq::observe_result(expVal.real(), ham, perTermData);
  }

#ifdef TENSORNET_FP32
  virtual std::string name() const override { return "tensornet-mps-fp32"; }
#else
  virtual std::string name() const override { return "tensornet-mps"; }
#endif
  CircuitSimulator *clone() override {
    thread_local static auto simulator = std::make_unique<SimulatorMPS>();
    return simulator.get();
  }

  void deallocateStateImpl() override {
    m_acceleratedMps.reset();
    m_pendingZeroQubits = 0;
    SimulatorTensorNetBase<ScalarType>::deallocateStateImpl();
  }

  void setToZeroState() override {
    if (m_pendingZeroQubits > 0)
      return;
    if (m_acceleratedMps) {
      m_acceleratedMps->resetZero();
      return;
    }
    SimulatorTensorNetBase<ScalarType>::setToZeroState();
  }

  void addQubitsToState(std::size_t numQubits, const void *ptr) override {
    if constexpr (accelerationAvailable) {
      if (!ptr && !m_state && accelerationConfigurationSupported()) {
        if (m_acceleratedMps)
          m_acceleratedMps->appendZeroQubits(numQubits);
        else
          m_pendingZeroQubits += numQubits;
        return;
      }
    }

    materializeAcceleratedState();
    LOG_API_TIME();
    if (!m_state) {
      if (!ptr) {
        m_state = std::make_unique<TensorNetState<ScalarType>>(
            numQubits, scratchPad, m_cutnHandle, m_randomEngine);
      } else {
        auto [state, mpsTensors] =
            MPSSimulationState<ScalarType>::createFromStateVec(
                m_cutnHandle, scratchPad, 1ULL << numQubits,
                reinterpret_cast<std::complex<ScalarType> *>(
                    const_cast<void *>(ptr)),
                m_settings.maxBond, m_randomEngine);
        m_state = std::move(state);
      }
    } else {
      if (!ptr) {
        auto tensors = m_state->factorizeMPS(
            m_settings.maxBond, m_settings.absCutoff, m_settings.relCutoff,
            m_settings.svdAlgo, m_settings.gaugeOption);
        // The right most MPS tensor needs to have one more extra leg (no longer
        // the boundary tensor).
        tensors.back().extents.emplace_back(1);
        // The newly added MPS tensors are in zero state
        constexpr std::complex<ScalarType> tensorBody[2]{1.0, 0.0};
        constexpr auto tensorSizeBytes = 2 * sizeof(std::complex<ScalarType>);
        for (std::size_t i = 0; i < numQubits; ++i) {
          const std::vector<int64_t> extents =
              (i != numQubits - 1) ? std::vector<int64_t>{1, 2, 1}
                                   : std::vector<int64_t>{1, 2};
          void *mpsTensor{nullptr};
          HANDLE_CUDA_ERROR(cudaMalloc(&mpsTensor, tensorSizeBytes));
          HANDLE_CUDA_ERROR(cudaMemcpy(mpsTensor, tensorBody, tensorSizeBytes,
                                       cudaMemcpyHostToDevice));
          tensors.emplace_back(MPSTensor(mpsTensor, extents));
        }
        m_state = TensorNetState<ScalarType>::createFromMpsTensors(
            tensors, scratchPad, m_cutnHandle, m_randomEngine);
      } else {
        // Non-zero state needs to be factorized and appended.
        auto [state, mpsTensors] =
            MPSSimulationState<ScalarType>::createFromStateVec(
                m_cutnHandle, scratchPad, 1ULL << numQubits,
                reinterpret_cast<std::complex<ScalarType> *>(
                    const_cast<void *>(ptr)),
                m_settings.maxBond, m_randomEngine);
        auto tensors = m_state->factorizeMPS(
            m_settings.maxBond, m_settings.absCutoff, m_settings.relCutoff,
            m_settings.svdAlgo, m_settings.gaugeOption);
        // Adjust the extents of the last tensor in the original state
        tensors.back().extents.emplace_back(1);

        // Adjust the extents of the first tensor in the state to be appended.
        auto extents = mpsTensors.front().extents;
        extents.insert(extents.begin(), 1);
        mpsTensors.front().extents = extents;
        // Combine the list
        tensors.insert(tensors.end(), mpsTensors.begin(), mpsTensors.end());
        m_state = TensorNetState<ScalarType>::createFromMpsTensors(
            tensors, scratchPad, m_cutnHandle, m_randomEngine);
      }
    }
  }

  std::unique_ptr<cudaq::SimulationState> getSimulationState() override {
    LOG_API_TIME();
    materializeAcceleratedState();

    if (!m_state || m_state->getNumQubits() == 0)
      return std::make_unique<MPSSimulationState<ScalarType>>(
          std::move(m_state), std::vector<MPSTensor>{}, scratchPad,
          m_cutnHandle, m_randomEngine);

    if (m_state->getNumQubits() > 1) {
      std::vector<MPSTensor> tensors = m_state->factorizeMPS(
          m_settings.maxBond, m_settings.absCutoff, m_settings.relCutoff,
          m_settings.svdAlgo, m_settings.gaugeOption);
      return std::make_unique<MPSSimulationState<ScalarType>>(
          std::move(m_state), tensors, scratchPad, m_cutnHandle,
          m_randomEngine);
    }

    auto [d_tensor, numElements] = m_state->contractStateVectorInternal({});
    assert(numElements == 2);
    MPSTensor stateTensor;
    stateTensor.deviceData = d_tensor;
    stateTensor.extents = {static_cast<int64_t>(numElements)};

    return std::make_unique<MPSSimulationState<ScalarType>>(
        std::move(m_state), std::vector<MPSTensor>{stateTensor}, scratchPad,
        m_cutnHandle, m_randomEngine);
  }

  std::unique_ptr<cudaq::SimulationState>
  createStateFromData(const cudaq::state_data &data) override {
    return std::make_unique<MPSSimulationState<ScalarType>>(
               nullptr, std::vector<MPSTensor>{}, scratchPad, m_cutnHandle,
               m_randomEngine)
        ->createFromData(data);
  }

  bool requireCacheWorkspace() const override { return false; }
  bool canHandleGeneralNoiseChannel() const override { return true; }
  virtual ~SimulatorMPS() noexcept {
    for (auto &tensor : m_mpsTensors_d) {
      HANDLE_CUDA_ERROR(cudaFree(tensor.deviceData));
    }
    m_mpsTensors_d.clear();
  }
};
} // end namespace nvqir
