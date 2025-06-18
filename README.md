# The LLVM Compiler Infrastructure – Custom RISC-V Pseudo Instruction (DOT)

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/llvm/llvm-project/badge)](https://securityscorecards.dev/viewer/?uri=github.com/llvm/llvm-project)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/8273/badge)](https://www.bestpractices.dev/projects/8273)
[![libc++](https://github.com/llvm/llvm-project/actions/workflows/libcxx-build-and-test.yaml/badge.svg?branch=main&event=schedule)](https://github.com/llvm/llvm-project/actions/workflows/libcxx-build-and-test.yaml?query=event%3Aschedule)

Welcome to my custom fork of the LLVM project!

This repository builds upon the official LLVM source tree and includes a custom extension for the RISC-V backend: a new pseudo-instruction called `DOT`, designed to accelerate scalar dot-product operations.

---

## 🔧 Custom Contribution: `__builtin_riscv_dot` and the DOT Pseudo-Instruction

This fork introduces support for a new RISC-V-specific intrinsic, `__builtin_riscv_dot`, implemented as a pseudo-instruction named `DOT`. The goal of this extension is to provide efficient, low-level support for computing scalar dot products between two integer arrays.

### Why `DOT`?

The dot product is a fundamental operation in:
- signal processing,
- machine learning (especially in dense layer multiplications),
- physics engines,
- and various scientific computing tasks.

Since this operation involves tight loops and accumulations, its performance is crucial. By introducing a specialized pseudo-instruction in the backend, we enable LLVM to generate optimized code paths tailored for such workloads.

---

## 🔬 Implementation Overview

The following components have been modified or extended:

- **Frontend**:  
  Introduced a new builtin: `__builtin_riscv_dot(int* A, int* B, int size)`  
  This builtin lowers into the custom `DOT` pseudo-instruction via intrinsic mapping.

- **TableGen (.td) Definitions**:  
  - `DOT` was added as a pseudo-instruction to `RISCVInstrInfo.td`.  
  - Instruction patterns were linked to enable custom DAG selection.

- **Instruction Selection DAG (ISelDAG)**:  
  - `RISCVISelDAGToDAG.cpp` was extended to recognize and expand the `DOT` node.  
  - Pattern-matching logic identifies the builtin and dispatches the correct expansion logic.

- **Pseudo Instruction Expansion**:  
  - Final instruction emission is done in `RISCVExpandPseudoInsts.cpp`.  
  - A custom loop-unrolled version of the dot product was implemented, using caller-saved registers only.  
  - Optimized for minimal overhead in post-register allocation stage.

- **Target Machine Configuration**:  
  - The `RISCVTargetMachine.cpp` was updated to route the newly introduced pseudo-instruction through the proper expansion phase.

---

## 📈 Performance Results

Performance was measured using test loops on the BeagleV-Fire board with large integer vectors. Compared to standard scalar C implementations:
- Over **5× speed-up** in `-O0` optimization mode.
- Maintains competitive results in `-O2` and `-O3` modes.
- Performance gain comes from reduced memory traffic, loop unrolling, and hand-tuned register usage.

---

## 🧪 Experimental Platform

- **Hardware**: BeagleV-Fire SBC (RISC-V, LPDDR4, eMMC)
- **Cross-compilation**: Custom Clang + RISC-V GCC (dual toolchain)
- **Setup challenges**:
  - Windows-based cross-compilation caused OOM and BSOD errors.
  - Resolved via Linux Mint dual-boot environment.
  - UART debugging and reflashing using Arduino as a USB-serial bridge.

---

## 🧭 Getting the Source Code and Building LLVM

Consult the original [LLVM Getting Started Guide](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm) for base instructions.  
This repository follows the same layout and build system (CMake with `ninja`/`make` support).

To enable the DOT pseudo-instruction:
```bash
mkdir build && cd build
cmake -G Ninja ../llvm \
  -DLLVM_TARGETS_TO_BUILD=RISCV \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_ENABLE_PROJECTS=clang
ninja
