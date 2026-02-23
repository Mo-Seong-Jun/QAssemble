---
hide:
  - navigation
  - toc
---

# QAssemble

<div class="hero-section" markdown>

<div class="hero-text" markdown>

# QAssemble

**A pure-Python quantum simulation package** for calculating electronic properties of materials using diagrammatic many-body perturbation theory.

Built entirely with the Python standard library and a minimal set of well-established scientific packages — no compiled extensions or domain-specific frameworks required.

[Get Started](index.md#quick-start){ .md-button .md-button--primary }
[View on GitHub](https://github.com/Mo-Seong-Jun/QAssemble){ .md-button }

</div>

</div>

---

## What is QAssemble?

QAssemble is intentionally implemented in **pure Python**, meaning:

<div class="grid cards" markdown>

-   :material-language-python: **Pure Python**

    ---

    No C/C++/Fortran extensions beyond what NumPy/SciPy already provide. No proprietary or hard-to-install domain-specific libraries.

-   :material-eye: **Readable & Hackable**

    ---

    Every algorithm is visible and modifiable. Easy to install, easy to extend, and easy to understand.

-   :material-package-variant: **Easy to Install**

    ---

    All dependencies are standard Python packages installable via `pip`. No additional compilers or external libraries needed.

-   :material-scale-balance: **Open Source**

    ---

    Full source code available on GitHub. Community contributions welcome.

</div>

---

## Methods & Features

<div class="grid cards" markdown>

-   :material-atom: **Electronic Structure Methods**

    ---

    - **Tight-Binding (TB)** — non-interacting band structure
    - **Hartree-Fock (HF)** — mean-field theory (restricted/unrestricted)
    - **GW Approximation (GW)** — many-body perturbation theory

    [:octicons-arrow-right-24: Theory](theory/greens-function.md)

-   :material-calculator-variant: **Advanced Numerics**

    ---

    - Discrete Lehmann Representation (DLR)
    - Dyson equation solver
    - k-space ↔ real-space Fourier transforms
    - High-frequency tail fitting

    [:octicons-arrow-right-24: DLR Details](theory/dlr.md)

-   :material-chemical-weapon: **Coulomb Interactions**

    ---

    - **Local**: Slater-Kanamori, Slater, Kanamori
    - **Non-local**: Ohno, Ohno-Yukawa, J-threading (JTH)

    [:octicons-arrow-right-24: GW Approximation](theory/gw-approximation.md)

-   :material-cpu-64-bit: **Parallelization & I/O**

    ---

    - MPI-parallelized via `mpi4py` (with serial fallback)
    - Python dict-based configuration
    - HDF5 data storage via `h5py`

    [:octicons-arrow-right-24: Reference](SerialModules.md)

</div>

---

## Dependencies

| Package | Purpose |
|---|---|
| [NumPy](https://numpy.org/) | Array operations and linear algebra |
| [SciPy](https://scipy.org/) | Eigensolvers, interpolation, special functions |
| [h5py](https://www.h5py.org/) | HDF5-based data storage |
| [mpi4py](https://mpi4py.readthedocs.io/) | MPI parallelization |
| [Matplotlib](https://matplotlib.org/) | Plotting |

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Mo-Seong-Jun/QAssemble.git
cd QAssemble
pip install numpy scipy h5py mpi4py matplotlib
```

---

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

=== "Serial"

    ```bash
    python3 src/QuantumAssemble.py
    ```

=== "Parallel (MPI)"

    ```bash
    mpirun -n <num_processors> python3 src/QuantumAssemble.py
    ```

---

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

---

## Documentation

<div class="grid cards" markdown>

-   :material-book-open-variant: **Theory**

    ---

    Background on Green's functions, Hartree-Fock, GW, and DLR.

    [:octicons-arrow-right-24: Theory Section](theory/greens-function.md)

-   :material-file-document-outline: **Serial Modules**

    ---

    Module-by-module reference for the serial implementation.

    [:octicons-arrow-right-24: Serial Modules](SerialModules.md)

-   :material-sitemap: **Class Diagram**

    ---

    Composition and dependency diagrams for the codebase.

    [:octicons-arrow-right-24: Class Diagram](SerialClassDiagram.md)

</div>
