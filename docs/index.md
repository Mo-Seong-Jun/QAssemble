# QAssemble

**QAssemble** is a pure-Python quantum simulation package for calculating electronic properties of materials using diagrammatic many-body perturbation theory. Built entirely with the Python standard library and a minimal set of well-established scientific packages — no compiled extensions or domain-specific frameworks required.

## Why Pure Python?

QAssemble is intentionally implemented in **pure Python**, meaning:

- No C/C++/Fortran extensions beyond what NumPy/SciPy already provide
- No proprietary or hard-to-install domain-specific libraries
- Readable, hackable source code — every algorithm is visible and modifiable
- Easy to install, easy to extend, and easy to understand

All dependencies are standard Python packages installable via `pip`. No additional compilers or external libraries are needed.

## Features

- **Methods**:
    - Tight-Binding (TB) — non-interacting band structure
    - Hartree-Fock (HF) — mean-field theory (restricted/unrestricted)
    - GW Approximation (GW) — many-body perturbation theory
- **Advanced Numerics**:
    - Discrete Lehmann Representation (DLR) for high-precision imaginary-time ↔ Matsubara frequency transforms
    - Dyson equation solver for renormalized Green's functions
    - k-space ↔ real-space Fourier transforms with phase-correct basis handling
    - High-frequency tail fitting for asymptotic accuracy
- **Coulomb Interactions**:
    - Local: Slater-Kanamori, Slater, Kanamori parameterizations
    - Non-local: Ohno, Ohno-Yukawa, J-threading (JTH)
- **Parallelization**: MPI-parallelized implementations via `mpi4py` (with graceful serial fallback)
- **Input/Output**: Python dict-based configuration and HDF5 data storage via `h5py`
- **Crystal Structure**: Lattice vectors, basis positions, k-point grids, spin-orbit coupling (SOC)

## Dependencies

| Package | Purpose |
|---|---|
| [NumPy](https://numpy.org/) | Array operations and linear algebra |
| [SciPy](https://scipy.org/) | Eigensolvers, interpolation, special functions |
| [h5py](https://www.h5py.org/) | HDF5-based data storage |
| [mpi4py](https://mpi4py.readthedocs.io/) | MPI parallelization |
| [Matplotlib](https://matplotlib.org/) | Plotting |

## Installation

Clone the repository:

```bash
git clone https://github.com/Mo-Seong-Jun/QAssemble.git
cd QAssemble
```

Install dependencies:

```bash
pip install numpy scipy h5py mpi4py matplotlib
```

## Quick Start

### 1. Prepare `input.ini`

The input file uses plain Python dict syntax:

```python
Crystal = {
    'RVec': [[1,0,0],[0.5,0.866,0],[0,0,1]],
    'SOC': False,
    'CorF': 'F',                          # 'F' = Fractional, 'C' = Cartesian
    'Basis': [[[0.33333,0.33333,0],1],
              [[0.66667,0.66667,0],1]],
    'NSpin': 1,
    'NElec': 2,
    'KGrid': [25,25,1]
}

Hamiltonian = {
    'OneBody': {
        'Hopping': {
            ((0,0),(1,0)): {
                1.0: [[0,0,0],[-1,0,0],[0,-1,0]],
            },
        },
        'Onsite': {
            0: {(0,0): 0.0, (1,0): 0.0}
        }
    },
    'TwoBody': {
        'Local': {
            'Parameter': 'SlaterKanamori',
            'option': {
                (0,(0)): {'l': 0, 'U': 2.0, 'Up': 0.0},
                (1,(0)): {'l': 0, 'U': 2.0, 'Up': 0.0}
            }
        },
        'NonLocal': {
            ((0,0),(1,0)): {
                0.20: [[0,0,0],[-1,0,0],[0,-1,0]],
            },
        }
    }
}

Control = {
    'Method': 'gw',           # 'tb', 'hf', or 'gw'
    'Prefix': 'my_calc',
    'NSCF': 20000,
    'Mix': 0.1,
    'T': 2000,
    'MatsubaraCutOff': 100,
    'ConstantW': 1.0
}
```

### 2. Run

Serial:

```bash
python3 src/QuantumAssemble.py
```

Parallel (MPI):

```bash
mpirun -n <num_processors> python3 src/QuantumAssemble.py
```

## Directory Structure

```
QAssemble/
├── src/
│   ├── QuantumAssemble.py      # Main entry point
│   └── QAssemble/
│       ├── MPI/                # MPI-parallelized implementations
│       ├── Serial/             # Serial implementations
│       └── utility/            # Shared numerics (DLR, Dyson, Fourier, etc.)
└── docs/                       # Documentation
```

For deeper documentation, see:

- [Serial Modules](SerialModules.md) — module-by-module reference
- [Class Diagram](SerialClassDiagram.md) — composition and dependency diagrams
