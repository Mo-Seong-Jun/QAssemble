# QAssemble

**QAssemble** is a Python-based quantum simulation package for calculating electronic properties of materials using diagrammatic expansion methods. This package supports Tight-Binding (TB), Hartree-Fock (HF), and GW Approximation (GW) solvers, and allows for parallel execution via `mpi4py`.

## Key Features

* **Physical Methodologies**: Support for Tight-Binding, Restricted/Unrestricted Hartree-Fock, and GW Approximation.
* **Parallelization**: Efficient parallel computing support utilizing `mpi4py`.
* **Input/Output**: Flexible configuration via `.ini` files and data storage using HDF5.
* **Hamiltonian Construction**: Support for hopping, onsite energy, Spin-Orbit Coupling (SOC), and various two-body interactions such as Slater-Kanamori and Ohno.

## Architecture Overview

QAssemble uses a modular structure where physical logic is decoupled from numerical implementation.

* **CorrelationFunction**: Orchestrates the entire workflow, constructing Crystal data and DLR frequency grids.
* **Fermionic Stack**: Responsible for Green's functions and self-energy calculations in (k, omega) space.
* **Bosonic Stack**: Manages bosonic response functions and screened interaction tensors.

## Installation and Getting Started

### Installation

```bash
git clone [https://github.com/sangkookchoi/QAssemble.git](https://github.com/sangkookchoi/DiagE.git)
cd DiagE
pip install numpy scipy h5py mpi4py matplotlib


### Running Simulations
* **Serial Execution** : '''bash 
